import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from light.nn import _FCNHead
def conv_1x1_bn(inp, oup, norm=True):
    if norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.SiLU()
            )
    else:
        return nn.Conv2d(inp, oup, 1, 1, 0, bias=False)


def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class XCA(nn.Module):
    """ 
        Cross-Covariance Attention (XCA) 
    """
    def __init__(self, embed_dim, qkv_dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_proj = nn.Linear(embed_dim, qkv_dim*3, bias=True)
        self.softmax = nn.Softmax(dim = -1)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(qkv_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = torch.matmul(attn, v)
        x = rearrange(x, 'b p h n d -> b p d (h n)')
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class XCAEncoder(nn.Module):
    def __init__(self, embed_dim, qkv_dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(embed_dim, XCA(embed_dim, qkv_dim, num_heads))
        self.ffn = PreNorm(embed_dim, FeedForward(embed_dim, mlp_dim, dropout))
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class XCABlock(nn.Module):
    def __init__(self,
                 channels, 
                 embed_dim, 
                 depth,
                 qkv_dim,
                 num_heads,
                 mlp_ratio=4., 
                 patch_size=(2,2)
                 ):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv_3x3_in = conv_3x3_bn(channels, channels)
        self.conv_1x1_in = conv_1x1_bn(channels, embed_dim, norm=False)

        mlp_dim = int(embed_dim * mlp_ratio)
        xca_transformer = nn.ModuleList([])
        for i in range(depth):
            xca_transformer.append(XCAEncoder(embed_dim, qkv_dim, num_heads, mlp_dim))
        xca_transformer.append( nn.LayerNorm(embed_dim))
        self.transformer = nn.Sequential(*xca_transformer)

        self.conv_1x1_out = conv_1x1_bn(embed_dim, channels)
        self.conv_3x3_out = conv_3x3_bn(2 * channels, channels)

    def forward(self, x):
        _, _, h, w = x.shape
        # make sure to height and width are divisible by patch size
        new_h = int(math.ceil(h / self.ph) * self.ph)
        new_w = int(math.ceil(w / self.pw) * self.pw)
        if new_h != h and new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        y = x.clone()
        # Local representations
        x = self.conv_3x3_in(x)
        x = self.conv_1x1_in(x)
        # Global representations
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=new_h//self.ph, w=new_w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = self.conv_1x1_out(x)
        x = torch.cat((x, y), 1)
        x = self.conv_3x3_out(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x)


class MV3Block(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size=3,
                 stride=1,
                 reduction_ratio=4,
                 drop_connect_rate=0.2
                 ):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [
                nn.Conv2d(in_planes, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                ]
        
        padding = self._get_padding(kernel_size, stride)
        layers += [
            nn.ZeroPad2d(padding),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=0, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)

# 1.43 M, 0.36 GMc
efficientvit_b0_cfg = {
    'qkv_dim': [64, 64, 64],
    'embed_dim': [64, 80, 96],
    'num_heads': 4,
    'depth': [2, 3, 4],
    'mlp_ratio': [2, 3, 4],
    'channels': [16, 16, 24, 48, 64, 80],
    'expansion': 2,
    'exp_factor': 4,    
} 

# 2.02 M, 0.60 GMc
efficientvit_b1_cfg = {
    'qkv_dim': [64, 64, 64],
    'embed_dim': [64, 96, 120],
    'num_heads': 4,
    'depth': [2, 3, 4],
    'mlp_ratio': [2, 3, 4],
    'channels': [16, 32, 48, 64, 80, 96],
    'expansion': 2,
    'exp_factor': 4,    
}

# 5.48 M, 1.73 GMc
efficientvit_b2_cfg = {
    'qkv_dim': [96, 96, 96],
    'embed_dim': [144, 192, 240],
    'num_heads': 4,
    'depth': [2, 3, 4],
    'mlp_ratio': [2, 2, 3],
    'channels': [16, 32, 64, 96, 128, 160],
    'expansion': 4,
    'exp_factor': 4,    
}


model_cfg = {
    'efficientvit_b0': efficientvit_b0_cfg, 
    'efficientvit_b1': efficientvit_b1_cfg, 
    'efficientvit_b2': efficientvit_b2_cfg, 
    }


class EfficientViT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        cfg = model_cfg[model_name]
        expansion = cfg['expansion']
        exp_factor = cfg['exp_factor']
        num_heads = cfg['num_heads']
        depth = cfg['depth']
        mlp_ratio = cfg['mlp_ratio']
        embed_dim = cfg['embed_dim']
        channels = cfg['channels']
        qkv_dim = cfg['qkv_dim']

        self.conv_in = conv_3x3_bn(3, channels[0], stride=2) # 128 x 128

        # MV3 Block
        self.layer1 = MV3Block(channels[0], channels[1], expansion, 3, 1) 

        layer2 = nn.ModuleList([])
        layer2.append(MV3Block(channels[1], channels[2], expansion, 3, 2)) # 64 x 64
        layer2.append(MV3Block(channels[2], channels[2], expansion, 5, 1))
        layer2.append(MV3Block(channels[2], channels[2], expansion, 5, 1))
        self.layer2 = nn.Sequential(*layer2)

        layer3 = nn.ModuleList([])
        layer3.append(MV3Block(channels[2], channels[3], expansion, 3, 2)) # 32 x 32
        layer3.append(XCABlock(channels[3],
                               embed_dim[0], 
                               depth[0],
                               qkv_dim[0],
                               num_heads,
                               mlp_ratio[0]))
        self.layer3 = nn.Sequential(*layer3)


        layer4 = nn.ModuleList([])
        layer4.append(MV3Block(channels[3], channels[4], expansion, 5, 2)) # 16 x 16
        layer4.append(XCABlock(channels[4],
                               embed_dim[1], 
                               depth[1],
                               qkv_dim[1],
                               num_heads,
                               mlp_ratio[1]))
        self.layer4 = nn.Sequential(*layer4)

        layer5 = nn.ModuleList([])
        layer5.append(MV3Block(channels[4], channels[5], expansion, 5, 2)) # 8 x 8
        layer5.append(XCABlock(channels[5],
                               embed_dim[2], 
                               depth[2],
                               qkv_dim[2], 
                               num_heads,
                               mlp_ratio[2]))
        self.layer5 = nn.Sequential(*layer5)

        self.conv_out = conv_1x1_bn(channels[5], channels[5] * exp_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        # self.classifier = nn.Linear(in_features=channels[5] * exp_factor, out_features=1000, bias=True)
        self.head = _FCNHead(96, 19, norm_kwargs=None)
        # self.classify_camvid = nn.Conv2d(640, 19, 1, 1, 0, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        x = self.conv_in(x)
        x = self.layer1(x)
        c1 = self.layer2(x)
        c2 = self.layer3(c1)
        c3 = self.layer4(c2)
        c4 = self.layer5(c3)

        outputs = list()
        x_ = self.head(c4)
        x_ = F.interpolate(x_, size, mode='bilinear', align_corners=True)
        outputs.append(x_)

        # x = self.conv_out(x)
        # x = self.classify_camvid(x)
        # x = nn.functional.interpolate(x, scale_factor=32, mode='bilinear')

        # x = self.pool(x).view(-1, x.shape[1])
        # x = self.classifier(x)
        return tuple(outputs)


if __name__ == '__main__':
    import torch
    x = torch.randn(5, 3, 224, 224)
    model = EfficientViT(model_name='efficientvit_b2')
    ckpt = torch.load('b2.pt', map_location='cpu')
    model.load_state_dict(ckpt, False)
    y = model(x)
    print(y.shape)











