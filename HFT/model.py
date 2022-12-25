import torch, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.nn.init as init
from modules import DeformConvPack


def space_to_depth(in_tensor, down_scale):
    n, c, h, w = in_tensor.size()
    unfolded_x = torch.nn.functional.unfold(in_tensor, down_scale, stride=down_scale)
    return unfolded_x.view(n, c * down_scale ** 2, h // down_scale, w // down_scale)

# 用于学习偏移量
class Create_Offest(nn.Module):
    def __init__(self, nFeat):
        super(Create_Offest, self).__init__()
        self.nFeat = nFeat
        self.offest1 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU(),
                                     nn.Conv2d(self.nFeat, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.offest2 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU(),
                                     nn.Conv2d(self.nFeat, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.offest3 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU(),
                                     nn.Conv2d(self.nFeat, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.offest4 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU(),
                                     nn.Conv2d(self.nFeat, self.nFeat, 3, 1, 1), nn.LeakyReLU())

    def forward(self, x1, x2, x3, x4):
        off_x1 = self.offest1(x1)
        off_x2 = self.offest2(x2)
        off_x3 = self.offest3(x3)
        off_x4 = self.offest4(x4)
        return off_x1, off_x2, off_x3, off_x4


class Create_Kernels(nn.Module):
    def __init__(self, nFeat):
        super(Create_Kernels, self).__init__()
        self.nFeat = nFeat
        self.kernels1 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.kernels2 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.kernels3 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.kernels4 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())

    def forward(self, x1, x2, x3, x4):
        ker_x1 = self.kernels1(x1)
        ker_x2 = self.kernels2(x2)
        ker_x3 = self.kernels3(x3)
        ker_x4 = self.kernels4(x4)
        return ker_x1, ker_x2, ker_x3, ker_x4



class Conv_model_multiscale(nn.Module):
    def __init__(self, in_chans, nFeat):
        super(Conv_model_multiscale, self).__init__()
        self.in_chans = in_chans
        self.nFeat = nFeat
        self.conv_lrelu1 = nn.Sequential(nn.Conv2d(self.in_chans, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_lrelu2 = nn.Sequential(nn.Conv2d(self.in_chans * 4, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_lrelu3 = nn.Sequential(nn.Conv2d(self.in_chans * 16, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_lrelu4 = nn.Sequential(nn.Conv2d(self.in_chans * 4 * 16, self.nFeat, 3, 1, 1), nn.LeakyReLU())

    def forward(self, x):
        x1 = space_to_depth(x, 2)
        x2 = space_to_depth(x1, 2)
        x3 = space_to_depth(x2, 2)
        t = self.conv_lrelu1(x)
        t1 = self.conv_lrelu2(x1)
        t2 = self.conv_lrelu3(x2)
        t3 = self.conv_lrelu4(x3)

        return t, t1, t2, t3


class Attention_Fusion_Block(nn.Module):
    def __init__(self):
        super(Attention_Fusion_Block, self).__init__()
        self.Upsample = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2_up = self.Upsample(x2)
        x_fuse = x1 * 0.5 + x2_up * 0.5
        return x_fuse


class Multiscale_attention_block(nn.Module):
    def __init__(self, nFeat):
        super(Multiscale_attention_block, self).__init__()
        self.nFeat = nFeat
        self.conv1 = nn.Conv2d(self.nFeat * 2, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(self.nFeat * 2, 1, 3, padding=1)
        self.conv3 = nn.Conv2d(self.nFeat * 2, 1, 3, padding=1)
        self.conv4 = nn.Conv2d(self.nFeat * 2, 1, 3, padding=1)
        self.a_f_b1 = Attention_Fusion_Block()
        self.a_f_b2 = Attention_Fusion_Block()
        self.a_f_b3 = Attention_Fusion_Block()

    def forward(self, f_x1, f1_x1, f2_x1, f3_x1,
                f_x2, f1_x2, f2_x2, f3_x2):
        f3 = torch.cat([f3_x1, f3_x2], 1)
        f2 = torch.cat([f2_x1, f2_x2], 1)
        f1 = torch.cat([f1_x1, f1_x2], 1)
        f = torch.cat([f_x1, f_x2], 1)

        att_f3 = self.conv1(f3)
        att_f3_x1x2 = torch.sigmoid(att_f3)
        f3_x1x2 = torch.mul(att_f3_x1x2, f3_x1)

        att_f2 = self.conv2(f2)
        att_f2 = self.a_f_b1(att_f2, att_f3)
        att_f2_x1_x2 = torch.sigmoid(att_f2)
        f2_x1x2 = torch.mul(att_f2_x1_x2, f2_x1)

        att_f1 = self.conv3(f1)
        att_f1 = self.a_f_b2(att_f1, att_f2)
        att_f1_x1_x2 = torch.sigmoid(att_f1)
        f1_x1x2 = torch.mul(att_f1_x1_x2, f1_x1)

        att_f = self.conv4(f)
        att_f = self.a_f_b3(att_f, att_f1)
        att_f_x1_x2 = torch.sigmoid(att_f)
        f_x1x2 = torch.mul(att_f_x1_x2, f_x1)

        return f_x1x2, f1_x1x2, f2_x1x2, f3_x1x2

# ----------------通道注意力----------------------
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y





def reverse_patches(images, out_size, ksizes, strides, padding):
    unfold = torch.nn.Fold(output_size = out_size,
                            kernel_size=ksizes,
                            dilation=1,
                            padding=padding,
                            stride=strides)
    patches = unfold(images)
    return patches

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class PatchEmbed(nn.Module):
    def __init__(self, img_size=48, patch_size=2, in_chans=64, embed_dim=768):
        super().__init__()
        img_size = tuple((img_size,img_size))
        patch_size = tuple((patch_size,patch_size))
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape #16*64*48*48
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # pdb.set_trace()
        x = self.proj(x).flatten(2).transpose(1, 2)#64*48*48->768*6*6->768*36->36*768
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features//4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.qkv = nn.Linear(dim//2, dim//2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim//2, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q_all = torch.split(q, math.ceil(N//4), dim=-2)
        k_all = torch.split(k, math.ceil(N//4), dim=-2)
        v_all = torch.split(v, math.ceil(N//4), dim=-2)
        output = []
        for q,k,v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale   #16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)#.reshape(B, N, C)

            output.append(trans_x)
        x = torch.cat(output,dim=1)
        x = x.reshape(B,N,C)
        x = self.proj(x)
        return x


# 没有小波变换没有隐士变换
class DRDB(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32):
        super(DRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.bp1_conv1 = nn.Conv2d(in_ch * 3 + in_ch // 2, in_ch, 3, padding=1)
        self.bp2_conv1 = nn.Conv2d(in_ch * 3 + in_ch // 2, in_ch, 3, padding=1)
        self.bp3_conv1 = nn.Conv2d(in_ch * 3 + in_ch // 2, in_ch, 3, padding=1)
        self.bp4_conv1 = nn.Conv2d(in_ch * 3 + in_ch // 2, in_ch, 3, padding=1)

        self.bp1_conv2 = nn.Conv2d(in_ch, in_ch // 4, 3, padding=1)
        self.bp2_conv2 = nn.Conv2d(in_ch, in_ch // 4, 3, padding=1)
        self.bp3_conv2 = nn.Conv2d(in_ch, in_ch // 4, 3, padding=1)
        self.bp4_conv2 = nn.Conv2d(in_ch, in_ch // 4, 3, padding=1)

        in_ch_ = in_ch
        self.Dcov1 = nn.Sequential(nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2), nn.LeakyReLU())
        in_ch_ += growth_rate
        self.Dcov2 = nn.Sequential(nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2), nn.LeakyReLU())
        in_ch_ += growth_rate
        self.Dcov3 = nn.Sequential(nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2), nn.LeakyReLU())
        in_ch_ += growth_rate
        self.Dcov4 = nn.Sequential(nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2), nn.LeakyReLU())
        in_ch_ += growth_rate
        self.Dcov5 = nn.Sequential(nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2), nn.LeakyReLU())
        in_ch_ += growth_rate
        self.conv = nn.Sequential(nn.Conv2d(in_ch_, in_ch, 1, padding=0), nn.LeakyReLU())

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.Dcov1(x0)
        x1 = torch.cat([x0, x1], dim=1)
        x2 = self.Dcov2(x1)
        x2 = torch.cat([x1, x2], dim=1)
        x3 = self.Dcov3(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.Dcov4(x3)
        x4 = torch.cat([x3, x4], dim=1)
        x5 = self.Dcov5(x4)
        x5 = torch.cat([x4, x5], dim=1)
        t1 = self.bp1_conv1(x5)
        t1 = self.bp1_conv2(t1)
        t2 = self.bp2_conv1(x5)
        t2 = self.bp1_conv2(t2)
        t3 = self.bp3_conv1(x5)
        t3 = self.bp3_conv2(t3)
        t4 = self.bp4_conv1(x5)
        t4 = self.bp4_conv2(t4)
        t = torch.cat([t1, t2, t3, t4], dim=1)

        out = x + t
        return out

class CADB(nn.Module):
    def __init__(self, in_chan, nBlocks):
        super(CADB, self).__init__()
        self.nBlocks = nBlocks
        models = []
        for i in range(nBlocks):
            models.append(CALayer(in_chan))
            models.append(DRDB(in_ch=in_chan))
        self.drdbs = nn.Sequential(*models)

    def forward(self, x):
        x = self.drdbs(x)
        return x

## Base block
class MLABlock(nn.Module):
    def __init__(
        self, n_feat = 64, drop=0., act_layer=nn.ReLU):
        super(MLABlock, self).__init__()
        self.dim = n_feat*9
        self.atten = EffAttention(self.dim, num_heads=8, qkv_bias=False, qk_scale=None,attn_drop=0., proj_drop=0.)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=n_feat*9, hidden_features=n_feat*9//4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)
    def forward(self, x):
        x = extract_image_patches(x, ksizes=[3, 3],
                                      strides=[1,1],
                                      rates=[1, 1],
                                      padding='same')#   16*2304*576
        x = x.permute(0,2,1)
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))#self.drop_path(self.mlp(self.norm2(x)))
        return x

class Al_PFM(nn.Module):
    def __init__(self, in_chans, nFeat):
        super(Al_PFM, self).__init__()
        self.in_chans = in_chans
        self.nFeat = nFeat

        self.first_con1 = nn.Sequential(nn.Conv2d(self.in_chans, self.nFeat // 2, 3, 1, 1), nn.LeakyReLU())
        self.first_con2 = nn.Sequential(nn.Conv2d(self.in_chans, self.nFeat // 2, 3, 1, 1), nn.LeakyReLU())
        self.first_con3 = nn.Sequential(nn.Conv2d(self.in_chans, self.nFeat // 2, 3, 1, 1), nn.LeakyReLU())

        self.offest1 = nn.Sequential(nn.Conv2d(self.nFeat, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.offest2 = nn.Sequential(nn.Conv2d(self.nFeat, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.offest3 = nn.Sequential(nn.Conv2d(self.nFeat, self.nFeat, 3, 1, 1), nn.LeakyReLU())

        self.cmm1 = Conv_model_multiscale(self.nFeat, self.nFeat)
        self.cmm2 = Conv_model_multiscale(self.nFeat, self.nFeat)
        self.cmm3 = Conv_model_multiscale(self.nFeat, self.nFeat)

        self.dcn1 = DeformConvPack(nFeat, nFeat, 3, 1, 1)
        self.dcn2 = DeformConvPack(nFeat, nFeat, 3, 1, 1)
        self.dcn3 = DeformConvPack(nFeat, nFeat, 3, 1, 1)

        self.conv_2n1 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_2n2 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_2n3 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_2n4 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_2n5 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_2n6 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_2n7 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_2n8 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())

        self.mp = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)
        self.conv_3n1 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_3n2 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_3n3 = nn.Sequential(nn.Conv2d(self.nFeat * 2, self.nFeat, 3, 1, 1), nn.LeakyReLU())

        self.dcn_3n1 = DeformConvPack(nFeat, nFeat, 3, 1, 1)
        self.dcn_3n2 = DeformConvPack(nFeat, nFeat, 3, 1, 1)
        self.dcn_3n3 = DeformConvPack(nFeat, nFeat, 3, 1, 1)
        self.conv_3n1 = nn.Sequential(nn.Conv2d(self.nFeat*3, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_3n2 = nn.Sequential(nn.Conv2d(self.nFeat*3, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_3n3 = nn.Sequential(nn.Conv2d(self.nFeat*3, self.nFeat, 3, 1, 1), nn.LeakyReLU())
        self.conv_3n4 = nn.Sequential(nn.Conv2d(self.nFeat*3, self.nFeat, 3, 1, 1), nn.LeakyReLU())

        self.ltb = MLABlock(n_feat=nFeat)

    def forward(self, x1, x2, x3):
        f_x1 = self.first_con1(x1)
        f_x2 = self.first_con1(x2)
        f_x3 = self.first_con1(x3)

        f_x1x2 = torch.cat([f_x1, f_x2], dim=1)
        f_x2x2 = torch.cat([f_x2, f_x2], dim=1)
        f_x3x2 = torch.cat([f_x3, f_x2], dim=1)

        off_x1x2 = self.offest1(f_x1x2)
        off_x2x2 = self.offest2(f_x2x2)
        off_x3x2 = self.offest3(f_x3x2)
        f_x1x2 = self.dcn1(f_x1x2, off_x1x2)
        f_x2x2 = self.dcn2(f_x2x2, off_x2x2)
        f_x3x2 = self.dcn3(f_x3x2, off_x3x2)

        l1_x1,l1_x2,l1_x3 = self.mp(f_x1x2),self.mp(f_x2x2),self.mp(f_x3x2)
        l2_x1, l2_x2, l2_x3 = self.mp(l1_x1),self.mp(l1_x2),self.mp(l1_x3)
        l3_x1,l3_x2,l3_x3 = self.mp(l2_x1),self.mp(l2_x2),self.mp(l2_x3)

        l3_x1x2,l3_x3x2 = torch.cat([l3_x1,l3_x2],dim=1),torch.cat([l3_x3,l3_x2],dim=1)
        l2_x1x2,l2_x3x2 = torch.cat([l2_x1,l2_x2],dim=1),torch.cat([l2_x3,l2_x2],dim=1)
        l1_x1x2,l1_x3x2 = torch.cat([l1_x1,l1_x2],dim=1),torch.cat([l1_x3,l1_x2],dim=1)
        _f_x1x2,_f_x3x2 = torch.cat([f_x1x2,f_x2x2],dim=1),torch.cat([f_x3x2,f_x2x2],dim=1)

        l3_x1x2 = self.conv_2n1(l3_x1x2)
        l3_x3x2 = self.conv_2n2(l3_x3x2)
        l3_x1,l3_x3 = l3_x1*l3_x1x2,l3_x3*l3_x3x2
        l3 = torch.cat([l3_x1,l3_x2,l3_x3],dim=1)
        l3 = self.conv_3n1(l3)
        h, w = l3.size(2), l3.size(3)
        f = self.ltb(l3)
        f = f.permute(0, 2, 1)
        f = reverse_patches(f, (h, w), (3, 3), 1, 1)
        #l3 = self.up(f)
        l3 = F.interpolate(f, size=(h*2,w*2), mode='bilinear', align_corners=True)

        l2_x1x2 = self.conv_2n3(l2_x1x2)
        l2_x3x2 = self.conv_2n4(l2_x3x2)
        l2_x1, l2_x3 = l2_x1 * l2_x1x2, l2_x3 * l2_x3x2
        l2 = torch.cat([l2_x1, l2_x2, l2_x3], dim=1)
        l2 = self.conv_3n2(l2)-l3
        l2 = self.dcn_3n1(l2,l3)
#        l2 = self.up(l2)
        l2 = F.interpolate(l2, size=(h * 4, w * 4), mode='bilinear', align_corners=True)

        l1_x1x2 = self.conv_2n5(l1_x1x2)
        l1_x3x2 = self.conv_2n6(l1_x3x2)
        l1_x1, l1_x3 = l1_x1 * l1_x1x2, l1_x3 * l1_x3x2
        l1 = torch.cat([l1_x1, l1_x2, l1_x3], dim=1)
        l1 = self.conv_3n3(l1)-l2
        l1 = self.dcn_3n2(l1, l2)
#        l1 = self.up(l1)
        l1 = F.interpolate(l1, size=(h * 8, w * 8), mode='bilinear', align_corners=True)

        _f_x1 = self.conv_2n7(_f_x1x2)
        _f_x3 = self.conv_2n8(_f_x3x2)
        f_x1,f_x3 = f_x1x2*_f_x1,f_x3x2*_f_x3
        f = torch.cat([f_x1,f_x2x2,f_x3],dim=1)
        f = self.conv_3n4(f)-l1
        f = self.dcn_3n3(f, l1)

        return f



# 加上通道注意力后的merge1，单个
class IRM(nn.Module):
    def __init__(self, nFeat, Blocks):
        super(IRM, self).__init__()
        self.nFeat = nFeat
        self.Blocks = Blocks

        self.cadb = CADB(nFeat, self.Blocks)
        self.conv4 = nn.Conv2d(self.nFeat, 3, 3, padding=1)

    def forward(self, f):
        f = self.cadb(f)
        f = self.conv4(f)
        y = torch.sigmoid(f)
        return y


class HFT(nn.Module):
    def __init__(self, in_chan, nFeat, block):
        super(HFT, self).__init__()
        self.algin = Al_PFM(in_chan, nFeat)
        self.merge = IRM(nFeat, block)

    def forward(self, x1, x2, x3):
        f= self.algin(x1, x2, x3)
        y = self.merge(f)
        return y
