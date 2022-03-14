from functools import partial

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import data_providers
from local_attention import LocalAttention

from reformer_pytorch import default, LSHAttention, FullQKAttention, merge_dims, split_at_index, \
    expand_dim, process_inputs_chunk

from mae_main.util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import mae_main.models_vit as models_vit

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
        # print(out.shape)

        # torch.Size([16, 196, 384])
        # torch.Size([16, 49, 768])
        return out


class LSHSelfAttention(nn.Module):
    def __init__(self, dim, oup, heads=8, image_size=None, bucket_size=98, n_hashes=8, causal=False, dim_head=32, attn_chunks=1,
                 random_rotations_per_head=False, attend_across_buckets=True, allow_duplicate_attention=True,
                 num_mem_kv=0, one_value_head=False, use_full_attn=False, full_attn_thres=None, return_attn=False,
                 post_attn_dropout=0., dropout=0., n_local_attn_heads=0, **kwargs):
        super().__init__()
        self.ih, self.iw = image_size
        self.oup = oup

        # print(str(self.ih)+"  "+str(self.iw))
        assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)

        self.v_head_repeats = (heads if one_value_head else 1)
        v_dim = dim_heads // self.v_head_repeats

        # print("asdasd:"+str(self.dim))
        # print("asdasd:" + str(self.oup))
        self.temp1 = nn.Linear(192, self.dim, bias=False)   # 改动
        self.temp2 = nn.Linear(384, self.dim, bias=False)   # 改动
        self.temp3 = nn.Linear(768, self.dim, bias=False)   # 改动

        self.toqk = nn.Linear(self.dim, dim_heads, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim_heads, oup)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal,
                                     random_rotations_per_head=random_rotations_per_head,
                                     attend_across_buckets=attend_across_buckets,
                                     allow_duplicate_attention=allow_duplicate_attention, return_attn=return_attn,
                                     dropout=dropout, **kwargs)
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(window_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True,
                                         look_forward=(1 if not causal else 0))

        self.callback = None

    def forward(self, x, keys=None, input_mask=None, input_attn_mask=None, context_mask=None, pos_emb=None, **kwargs):
        device, dtype = x.device, x.dtype
        b, t, e, h, dh, m, l_h = *x.shape, self.heads, self.dim_head, self.num_mem_kv, self.n_local_attn_heads

        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(b, m, -1)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = torch.cat((x, mem, keys), dim=1)

        # print("1 x:"+str(x.shape))

        if x.shape[2] == 192:
            x = self.temp1(x)
        elif x.shape[2] == 768:
            x = self.temp3(x)
        else:
            x = self.temp2(x)

        qk = self.toqk(x)

        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        merge_batch_and_heads = partial(merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))

        has_local = l_h > 0
        lsh_h = h - l_h

        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))

        # print("2 v:" + str(x.shape))

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask

        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len=t, pos_emb=pos_emb, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks=self.attn_chunks)
        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)
        # print("3 out :"+str(out.shape))

        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))

        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)

        out = split_heads(out).view(b, t, -1)

        out = self.to_out(out)  # 改动
        # print("4 out:"+str(out.shape))
        return self.post_attn_dropout(out)


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        self.ih, self.iw = image_size
        self.downsample = downsample

        self.layers = []

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        dim = heads * dim_head
        # self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        #
        self.attn = LSHSelfAttention(dim, oup, dim_head=dim_head, image_size=image_size, heads=heads, dropout=dropout)

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

# def Vision_Transformer():
#     global_pool = True
#     model = models_vit.__dict__['vit_base_patch16'](
#         num_classes=100,
#         drop_path_rate=0.1,
#         global_pool=True,
#     )
#     checkpoint = torch.load('mae_pretrain_vit_base.pth', map_location='cpu')
#     checkpoint_model = checkpoint['model']
#     state_dict = model.state_dict()
#     for k in ['head.weight', 'head.bias']:
#         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]
#     interpolate_pos_embed(model, checkpoint_model)
#     # load pre-trained model
#     msg = model.load_state_dict(checkpoint_model, strict=False)
#     print(msg)
#     if global_pool:
#         assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
#     else:
#         assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
#     # manually initialize fc layer
#     trunc_normal_(model.head.weight, std=2e-5)
#     for n, p in model.named_parameters():
#         if n not in ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']:
#             p.requires_grad = False
#     return model

class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000,
                 block_types=['C', 'C', 'T', 'T','ViT']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}#, 'ViT': Vision_Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih//2 , iw//2 ))

        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih//4, iw//4 ))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih//8 , iw//8 ))
        #self.s3 = Vision_Transformer()
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
                # print("true")
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                # print("false")
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

    def _make_efficient_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])

        for i in range(depth):
            layers.append(block(inp, oup, image_size))
        return nn.Sequential(*layers)


def coatnet_0(types):
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=100,block_types=types)


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
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=100)

def coat_new():
    num_blocks = [2, 2, 12, 28, 2]  # L
    channels = [192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=100)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(epoch, train_loader, device, optimizer, model, criterion,model_name):
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
            name = model_name + 'network_params.pth'
            torch.save(model.state_dict(), name)
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

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('using device: {}'.format(device))
    block_types = ['C', 'C', 'C', 'T']
    model_name = 'CCCT_batah_size8'
    model = coatnet_0(block_types)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # computes softmax and then the cross entropy
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     # filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=0.005,
    #     weight_decay=0.05
    #     )  # 冲量
    optimizer = torch.optim.Adam(
        model.parameters(),
        # filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.005
    )  # 冲量

    print("Start training~~~")
    for epoch in range(100):
        train(epoch, train_loader, device, optimizer, model, criterion,model_name=model_name)
        test(epoch, val_loader, device, model, criterion)
    torch.save(model,'/models/coat_new.pt')

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
