import torch

class LsqQuantize(torch.autograd.Function):
    """
    LSQ: Learned Step Size Quantization (对称 INT8)
    前向：y = round(clamp(x / s, -Qmax, Qmax)) * s
    反向：对 x 使用 STE，对 s 使用 LSQ 梯度
    """
    @staticmethod
    def forward(ctx, x, s, Qmax):
        ctx.Qmax = Qmax
        x_scaled = x / s
        x_clamped = torch.clamp(x_scaled, -Qmax, Qmax)
        x_rounded = torch.round(x_clamped)
        y = x_rounded * s
        ctx.save_for_backward(x_scaled, x_rounded)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_scaled, x_rounded = ctx.saved_tensors
        Qmax = ctx.Qmax

        # 对输入 x 的梯度：STE
        grad_x = grad_output * (x_scaled.abs() <= Qmax).float()

        # 对 scale s 的梯度（LSQ 公式）
        # 在量化区间内：-x_scaled + x_rounded
        # 区间外：sign(x_scaled) * Qmax
        grad_s_inside = (-x_scaled + x_rounded) * (x_scaled.abs() <= Qmax).float()
        grad_s_outside = torch.where(x_scaled > Qmax, Qmax,
                                     torch.where(x_scaled < -Qmax, -Qmax,
                                                 torch.zeros_like(x_scaled)))
        grad_s = grad_s_inside + grad_s_outside
        # 乘以 grad_output，并对所有元素求和（s 是标量）
        grad_s = (grad_s * grad_output).sum()
        return grad_x, grad_s, None