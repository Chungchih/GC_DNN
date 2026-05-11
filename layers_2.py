import torch
import torch.nn as nn
import torch.nn.functional as F
from lsq_quant import LsqQuantize

class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, Qmax=127, noise_std_lsb=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.Qmax = Qmax
        self.noise_std_lsb = noise_std_lsb

        # 可学习的 scale 参数（先给占位值，随后会被初始化）
        self.s_x   = nn.Parameter(torch.tensor(1.0))
        self.s_w   = nn.Parameter(torch.tensor(1.0))
        self.s_out = nn.Parameter(torch.tensor(1.0))

        # 浮点权重与偏置
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.05)
        self.bias   = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 用初始权重初始化 s_w
        with torch.no_grad():
            init_sw = self.weight.abs().max() / self.Qmax
            self.s_w.data.fill_(init_sw)

        # 标记 s_x 和 s_out 是否已被初始化
        self.register_buffer('sx_initialized', torch.tensor(0))
        self.register_buffer('sout_initialized', torch.tensor(0))

    def forward(self, x):
        # 初始化 s_x（基于当前 batch）
        if self.training and self.sx_initialized == 0:
            with torch.no_grad():
                init_sx = x.detach().abs().max() / self.Qmax
                self.s_x.data.fill_(init_sx)
                self.sx_initialized.fill_(1)

        # 量化操作
        x_q = LsqQuantize.apply(x, self.s_x, self.Qmax)
        w_q = LsqQuantize.apply(self.weight, self.s_w, self.Qmax)
        y = F.conv2d(x_q, w_q, self.bias, stride=self.stride, padding=self.padding)

        # 初始化 s_out（基于第一次输出）
        if self.training and self.sout_initialized == 0:
            with torch.no_grad():
                init_sout = y.detach().abs().max() / self.Qmax
                self.s_out.data.fill_(init_sout)
                self.sout_initialized.fill_(1)

        y_q = LsqQuantize.apply(y, self.s_out, self.Qmax)

        if self.noise_std_lsb > 0 and self.training:
            noise = torch.randn_like(y_q) * self.noise_std_lsb * self.s_out
            y_q = y_q + noise
        return y_q

class QLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, Qmax=127, noise_std_lsb=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Qmax = Qmax
        self.noise_std_lsb = noise_std_lsb

        self.s_x   = nn.Parameter(torch.tensor(1.0))
        self.s_w   = nn.Parameter(torch.tensor(1.0))
        self.s_out = nn.Parameter(torch.tensor(1.0))

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        with torch.no_grad():
            init_sw = self.weight.abs().max() / self.Qmax
            self.s_w.data.fill_(init_sw)

        self.register_buffer('sx_initialized', torch.tensor(0))
        self.register_buffer('sout_initialized', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sx_initialized == 0:
            with torch.no_grad():
                init_sx = x.detach().abs().max() / self.Qmax
                self.s_x.data.fill_(init_sx)
                self.sx_initialized.fill_(1)

        x_q = LsqQuantize.apply(x, self.s_x, self.Qmax)
        w_q = LsqQuantize.apply(self.weight, self.s_w, self.Qmax)
        y = F.linear(x_q, w_q, self.bias)

        if self.training and self.sout_initialized == 0:
            with torch.no_grad():
                init_sout = y.detach().abs().max() / self.Qmax
                self.s_out.data.fill_(init_sout)
                self.sout_initialized.fill_(1)

        y_q = LsqQuantize.apply(y, self.s_out, self.Qmax)

        if self.noise_std_lsb > 0 and self.training:
            noise = torch.randn_like(y_q) * self.noise_std_lsb * self.s_out
            y_q = y_q + noise
        return y_q