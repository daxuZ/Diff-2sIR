# 导入所需的模块和函数
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

# 尝试导入 causal_conv1d_fn 和 causal_conv1d_update，如果导入失败则设为 None
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

# 尝试导入 selective_state_update，如果导入失败则设为 None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

# 尝试导入 RMSNorm、layer_norm_fn 和 rms_norm_fn，如果导入失败则设为 None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,                  # 模型维度
        d_state=16,               # 状态维度
        d_conv=4,                 # 卷积核大小
        expand=2,                 # 扩展倍数
        dt_rank="auto",           # dt_rank 的值，默认自动计算
        dt_min=0.001,             # dt 的最小值
        dt_max=0.1,               # dt 的最大值
        dt_init="random",         # dt 初始化方式
        dt_scale=1.0,             # dt 缩放因子
        dt_init_floor=1e-4,       # dt 初始化下限
        conv_bias=True,           # 卷积层是否使用偏置
        bias=False,               # 线性层是否使用偏置
        use_fast_path=True,       # 是否使用快速路径
        layer_idx=None,           # 层索引
        device=None,              # 设备（如 'cpu' 或 'cuda'）
        dtype=None,               # 数据类型（如 torch.float32）
    ):
        factory_kwargs = {"device": device, "dtype": dtype}  # 设备和数据类型参数
        super().__init__()  # 调用父类的初始化方法
        self.d_model = d_model  # 设置模型维度
        self.d_state = d_state  # 设置状态维度
        self.d_conv = d_conv  # 设置卷积核大小
        self.expand = expand  # 设置扩展倍数
        self.d_inner = int(self.expand * self.d_model)  # 计算内部维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 计算或设置 dt_rank
        self.use_fast_path = use_fast_path  # 设置是否使用快速路径
        self.layer_idx = layer_idx  # 设置层索引

        # 输入投影层，将输入维度转换为内部维度的两倍
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 1D卷积层，处理序列数据
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"  # 设置激活函数类型
        self.act = nn.SiLU()  # 定义 SiLU 激活函数

        # 定义 x_proj 和 dt_proj 线性层
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # 初始化 dt_proj 的权重
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化 dt_proj 的偏置，使得 F.softplus(dt_bias) 在 dt_min 和 dt_max 之间
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # softplus 的反函数
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # 标记 dt_proj.bias 不进行重新初始化

        # 初始化 S4D 的真实值
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # 保持 A_log 为浮点32位
        self.A_log = nn.Parameter(A_log)  # 定义 A_log 为可训练参数
        self.A_log._no_weight_decay = True  # 标记 A_log 不进行权重衰减

        # 定义跳过连接的参数 D
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # 保持为浮点32位
        self.D._no_weight_decay = True  # 标记 D 不进行权重衰减

        # 输出投影层，将内部维度转换回模型维度
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        前向传播函数
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape  # 获取批量大小、序列长度和维度

        conv_state, ssm_state = None, None  # 初始化卷积状态和SSM状态
        if inference_params is not None:  # 如果存在推理参数
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)  # 从缓存中获取状态
            if inference_params.seqlen_offset > 0:  # 如果序列偏移量大于0
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)  # 执行单步处理
                return out  # 返回输出

        # 矩阵乘法和转置操作
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")  # 加上偏置

        A = -torch.exp(self.A_log.float())  # 计算 A 的负指数值
        # 在反向传播过程中，我们将 dx 和 dz 放在一起，以避免 torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # 如果使用快速路径且推理参数为空
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # 输入相关的 B
                None,  # 输入相关的 C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)  # 将 xz 分成 x 和 z 两部分
            # 计算短卷积
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # 更新卷积状态
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])  # 使用激活函数处理卷积输出
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # 调整布局以避免额外的转置操作
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # 投影 x
            dA, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)  # 分割 x_dbl
            dA = rearrange(self.dt_proj(dA), "(b l) d -> b d l", l=seqlen)  # 投影 dA
            dA = torch.logaddexp(dA.float(), self.dt_proj.bias.float()[None, :, None]).to(x.dtype)  # 计算 dA
            B = rearrange(B, "(b l) m -> b m l", l=seqlen)  # 调整 B 的布局
            C = rearrange(C, "(b l) m -> b m l", l=seqlen)  # 调整 C 的布局

            # 初始化 Z，避免冗余分配
            if ssm_state is not None:
                Z = ssm_state
            else:
                Z = torch.zeros(B.shape, dtype=x.dtype, device=x.device)  # 初始化 Z

            # 计算离散化 A
            A = rearrange(A, "b m -> b m 1")  # 调整 A 的布局
            dA = rearrange(dA, "b m -> b m 1")  # 调整 dA 的布局
            D = rearrange(self.D, "m -> 1 m 1")  # 调整 D 的布局

            # 循环计算每一步
            for i in range(seqlen):
                # 扩展时钟
                dt = F.softplus(dA[:, :, i])  # 计算正的 dt

                # 幂运算计算 Z
                Z = Z + dt * (B[:, :, i] + Z * A)  # 更新 Z
                z[:, i] = Z * D + C[:, :, i]  # 计算 z

                if ssm_state is not None:
                    ssm_state[..., i] = Z  # 更新 ssm 状态

            # 合并 x 和 z，避免冗余分配
            out = self.out_proj(rearrange(z, "b d l -> l (b d)"))
            out = rearrange(out, "l (b d) -> b l d", b=batch)
        return out  # 返回最终输出
