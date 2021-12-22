import torch


class RevGrad(torch.autograd.Function):
    """Utility class to reverse gradient in case of DANN."""
    @staticmethod
    def forward(ctx, input_, scale_grad_down_factor):
        ctx.save_for_backward(input_)
        ctx.scale_grad_down_factor = scale_grad_down_factor
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -ctx.scale_grad_down_factor * grad_output
        return grad_input, None
