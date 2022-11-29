#
# ----------------------------------------------
import torch

class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, lambd=1.0):
    GradientReverse.scale = lambd
    return GradientReverse.apply(x)
