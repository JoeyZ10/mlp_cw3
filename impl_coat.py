# -*- coding: UTF-8 -*-
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
# from timm.models.vision_transformer import VisionTransformer
# from timm.models.layers import trunc_normal_
# from PyT_learning.mlp import data_providers
import data_providers as data_providers


total_train_accuracy = []
total_val_accuracy = []

total_train_loss = []
total_val_loss = []


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


# class DistilledVisionTransformer(VisionTransformer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
#         self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
#
#         trunc_normal_(self.dist_token, std=.02)
#         trunc_normal_(self.pos_embed, std=.02)
#         self.head_dist.apply(self._init_weights)
#
#     def forward_features(self, x):
#         # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#         # with slight modifications to add the dist_token
#         B = x.shape[0]
#         x = self.patch_embed(x)
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         dist_token = self.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         x = self.norm(x)
#         return x[:, 0], x[:, 1]
#
#     def forward(self, x):
#         x, x_dist = self.forward_features(x)
#         x = self.head(x)
#         x_dist = self.head_dist(x_dist)
#         if self.training:
#             return x, x_dist
#         else:
#             # during inference, return the average of both classifier predictions
#             return (x + x_dist) / 2


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000,
                 block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=100)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=100)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [128, 128, 256, 512, 1026]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=100)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=100)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(epoch, train_loader, device, optimizer, model, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    for index, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        target = target.long()
        target = target.cuda()

        optimizer.zero_grad()

        outputs = model.forward(inputs)

        _, predicted = torch.max(outputs.data, dim=1)  # predicted is the index for max value

        total += target.size(0)
        correct += (predicted == target).sum().item()

        target = target.long()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if index % 300 == 0:
            print('Train-------> [%d, %5d] loss: %.3f' % (epoch + 1, index + 1, running_loss / 300))
            print("Train-------> Accuracy: %d %%" % (100 * correct / total))
            running_loss = 0.0

    total_train_accuracy.append(correct / total)
    total_train_loss.append(running_loss)


#
#
def test(epoch, val_loader, device, model, criterion):
    correct = 0
    total = 0
    with torch.no_grad():
        for index, data in enumerate(val_loader, 0):
            images, labels = data

            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            labels = labels.cuda()

            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, dim=1)  # predicted is the index for max value

            loss = criterion(outputs, labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test--------> [%d, %5d] loss: %.3f' % (epoch + 1, index + 1, loss))
    print("Test--------> Accuracy: %d %%" % (100 * correct / total))
    print("\n")
    total_val_accuracy.append(correct / total)
    total_val_loss.append(loss)


if __name__ == '__main__':
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = data_providers.CIFAR100(root='data', set_name='train',
                                         transform=transform_train,
                                         download=True)  # initialize our rngs using the argument set seed
    val_data = data_providers.CIFAR100(root='data', set_name='val',
                                       transform=transform_test,
                                       download=True)  # initialize our rngs using the argument set seed
    test_data = data_providers.CIFAR100(root='data', set_name='test',
                                        transform=transform_test,
                                        download=True)  # initialize our rngs using the argument set seed

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('using device: {}'.format(device))

    model = coatnet_0()
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # computes softmax and then the cross entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.5) 

    print("Start training~~~")
    for epoch in range(100):
        train(epoch, train_loader, device, optimizer, model, criterion)
        test(epoch, val_loader, device, model, criterion)

    plt.clf()

    fig, axs = plt.subplots(2, 1, figsize=(10, 16), constrained_layout=True)

    # from PyT_learning.samples import total_train_accuracy, total_val_accuracy

    total_train_accuracy = torch.Tensor(total_train_accuracy).cpu()
    total_val_accuracy = torch.Tensor(total_val_accuracy).cpu()
    total_train_loss = torch.Tensor(total_train_loss).cpu()
    total_val_loss = torch.Tensor(total_val_loss).cpu()

    axs[0].plot(total_train_accuracy, 'r-', label='train accuracy')
    axs[0].plot(total_val_accuracy, 'b-', label='val accuracy')
    plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')

    axs[1].plot(total_train_loss, 'r-', label='train loss')
    axs[1].plot(total_val_loss, 'b-', label='val loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')

    plt.show()
    # img = torch.randn(1, 3, 224, 224)
    #
    # net = coatnet_0()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_1()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_2()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_3()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_4()
    # out = net(img)
    # print(out.shape, count_parameters(net))
