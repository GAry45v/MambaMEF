import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from pathlib import Path
from .AdaptivePEC import AdaptivePEC


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad

    return hook

class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96,patch_size=1):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, patch_size, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, patch_size, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    """1*1 conv before the output"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def window_partition(x, window_size):
    """
    Args:
    x: (B, C, H, W)
    window_size: window size
    Returns:
    local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    # 将图像分块
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
    windows: local window features (num_windows*B, window_size, window_size, C)
    window_size: Window size
    H: Height of image
    W: Width of image
    Returns:
    x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x

class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
    def _process_sequence(self, current_hidden_states):
        _B, _L, _D = current_hidden_states.shape # _D is d_model
        
        # Project to d_inner and split into xz_parts for convolution
        xz_projected = self.in_proj(current_hidden_states) # (B, L, d_inner)
        xz_projected_conv_input = rearrange(xz_projected, "b l d -> b d l") # (B, d_inner, L)
        x_part, z_part = xz_projected_conv_input.chunk(2, dim=1) # Each (B, d_inner/2, L)

        # Apply 1D convolution and SiLU to x_part
        x_conv = F.silu(F.conv1d(input=x_part, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                                 groups=self.d_inner // 2)) # (B, d_inner/2, L)
        
        # Apply 1D convolution and SiLU to z_part
        z_conv = F.silu(F.conv1d(input=z_part, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                                 groups=self.d_inner // 2)) # (B, d_inner/2, L)

        # Prepare inputs for selective_scan_fn from x_conv
        x_dbl = self.x_proj(rearrange(x_conv, "b d l -> (b l) d")) # ((B*L), dt_rank + d_state*2)
        
        dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=_L) # (B, d_inner/2, L)
        B_param = rearrange(B_param, "(b l) dstate -> b dstate l", l=_L).contiguous() # (B, d_state, L)
        C_param = rearrange(C_param, "(b l) dstate -> b dstate l", l=_L).contiguous() # (B, d_state, L)
        
        A_param = -torch.exp(self.A_log.float()) # (d_inner/2, d_state)

        ssm_out = selective_scan_fn(
            x_conv, dt, A_param, B_param, C_param, self.D.float(),
            z=None, 
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None
        ) # (B, d_inner/2, L)

        y_concat = torch.cat([ssm_out, z_conv], dim=1) # (B, d_inner, L)
        y_rearranged = rearrange(y_concat, "b d l -> b l d") # (B, L, d_inner)
        
        return y_rearranged

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, seqlen, D = hidden_states.shape
        
        # Assuming L is a perfect square, side = sqrt(L) (e.g., window_size)
        side = int(math.sqrt(seqlen))
        if side * side != seqlen:
            # Fallback or error for non-square L if strictly 2D interpretation is needed
            # For now, we proceed, but flips might be less meaningful if not square.
            # This case should ideally not happen if L comes from window_size*window_size
            pass

        # 1. Row-wise forward scan (original direction)
        y_row_fwd = self._process_sequence(hidden_states)

        # 2. Row-wise backward scan
        hidden_states_rev = torch.flip(hidden_states, dims=[1])
        y_row_bwd_rev = self._process_sequence(hidden_states_rev)
        y_row_bwd = torch.flip(y_row_bwd_rev, dims=[1])

        # Prepare for column-wise scans by transposing the spatial dimensions within the sequence
        # Reshape to 2D within sequence: (B, H, W, D) where H=W=side
        hidden_states_2d = rearrange(hidden_states, "b (h w) d -> b h w d", h=side, w=side)
        # Transpose H and W: (B, W, H, D)
        hidden_states_transposed_2d = torch.permute(hidden_states_2d, (0, 2, 1, 3))
        # Flatten back to sequence for column-wise scan: (B, L, D)
        hidden_states_colwise_flat = rearrange(hidden_states_transposed_2d, "b w h d -> b (w h) d")

        # 3. Column-wise forward scan (on transposed data)
        y_col_fwd_flat = self._process_sequence(hidden_states_colwise_flat)
        # Un-transpose the output
        y_col_fwd_transposed_2d = rearrange(y_col_fwd_flat, "b (w h) d -> b w h d", w=side, h=side)
        y_col_fwd_2d = torch.permute(y_col_fwd_transposed_2d, (0, 2, 1, 3)) # (B, H, W, D)
        y_col_fwd = rearrange(y_col_fwd_2d, "b h w d -> b (h w) d")

        # 4. Column-wise backward scan (on transposed data)
        hidden_states_colwise_flat_rev = torch.flip(hidden_states_colwise_flat, dims=[1])
        y_col_bwd_flat_rev = self._process_sequence(hidden_states_colwise_flat_rev)
        # Un-flip sequence order
        y_col_bwd_flat_unrev = torch.flip(y_col_bwd_flat_rev, dims=[1])
        # Un-transpose the output
        y_col_bwd_transposed_2d = rearrange(y_col_bwd_flat_unrev, "b (w h) d -> b w h d", w=side, h=side)
        y_col_bwd_2d = torch.permute(y_col_bwd_transposed_2d, (0, 2, 1, 3)) # (B, H, W, D)
        y_col_bwd = rearrange(y_col_bwd_2d, "b h w d -> b (h w) d")
        
        # Combine the outputs from the four directions
        # Each y_... has shape (B, L, d_inner)
        y_combined = y_row_fwd + y_row_bwd + y_col_fwd + y_col_bwd
        # Optionally, average them: y_combined = y_combined / 4.0
        
        out = self.out_proj(y_combined) # out_proj maps d_inner to d_model
        return out


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks, mlp_ratio=4., qkv_bias=False, qk_scale=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Mlp_block=Mlp,
                 layer_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # 如果是 transformer block
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 确保 Mlp_block 是 Mlp 类，或者传递的其他模块
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        # 打印输出形状调试
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer
    """

    def __init__(self, dim, depth, num_heads, window_size, conv=False, downsample=False, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., layer_scale=None, layer_scale_conv=None,
                 transformer_blocks=[]):
        super().__init__()
        self.conv = conv
        self.transformer_block = True

        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv) for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([Block(dim=dim, counter=i, transformer_blocks=transformer_blocks,
                                               num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale) for i in range(depth)])
        self.downsample = None if not downsample else Downsample(dim=dim)
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            # Padding to make sure the image is divisible by window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size

            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)

        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

        if self.downsample is None:
            return x
        return self.downsample(x)


class Encoder_Mamba(nn.Module):
    def __init__(self,
            dim=128,
            in_dim=64,
            in_chans=1,
            depth=2,  # 层数设置为12
            num_heads=4,
            window_size=8,  # 假设窗口大小为8
            conv=False,
            downsample=False,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.1,
            attn_drop=0.1,
            drop_path=0.1,
            layer_scale=None,
            layer_scale_conv=None,
            transformer_blocks=[]):
        super(Encoder_Mamba, self).__init__()
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(16+dim, 128)
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        self.MambaVision = MambaVisionLayer(
            dim=dim,
            depth=depth,  # 层数设置为12
            num_heads=num_heads,
            window_size=window_size,  # 假设窗口大小为8
            conv=conv,
            downsample=downsample,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            layer_scale=layer_scale,
            layer_scale_conv=layer_scale_conv,
            transformer_blocks=transformer_blocks
        )

    def forward(self, x):
        x_e = self.inc(x)
        x_t = self.patch_embed(x)
        x_t = self.MambaVision(x_t)
        x = torch.cat((x_e, x_t), dim=1)
        x = self.layer1(x)
        return x


class Decoder(nn.Module):
    """reconstruction"""

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer5 = DoubleConv(128,64)
        self.layer4 = DoubleConv(64, 32)
        self.layer3 = DoubleConv(32, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        output = self.outc(x)
        return output


class MambaMEF(nn.Module):
    """U-based network for self-reconstruction task"""

    def __init__(self):
        super(MambaMEF, self).__init__()
        # self.enhance1 = AdaptivePEC()
        # self.enhance2 = AdaptivePEC()
        self.encoder1 = Encoder_Mamba()
        self.encoder2 = Encoder_Mamba()
        self.decoder = Decoder()

    def forward(self, x1, x2):
        # x1 = self.enhance1(x1)
        # x2 = self.enhance2(x2)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = (x1+x2)/2
        x_fused = self.decoder(x)
        return x_fused

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MambaMEF().to(device)
    x1 = torch.rand(1, 1, 256, 256).to(device)
    x2 = torch.rand(1, 1, 256, 256).to(device)
    x = model(x1,x2)
    print(x.shape)