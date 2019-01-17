import torch


def norm_lorentz(x):
    return torch.sqrt(scalar_lorentz(x, x))


def scalar_lorentz(x, y):
    "Lorentz scalar product"
    r = x * y  # BD, BD -> BD
    return r[:, 1:].sum(dim=1) - r[:, 0]


def arcosh(x):
    return torch.log(x + torch.sqrt(x**2 - 1))


def distance_lorentz(x, y):
    return arcosh(-scalar_lorentz(x, y))


def set_x0(x):
    norm = torch.norm(x[:, 1:], dim=-1)
    print(norm)
    x[:, 0] = torch.sqrt(1 + norm)
    return x


def exp_(x, v):
    norm_v = norm_lorentz(v)
    return (torch.cosh(norm_v) * x) + torch.sinh(norm_v) * v / norm_v


def grad_lorentz(x):
    gl = torch.eye(x.size()[0])
    gl[0, 0] = -1
    h = gl * x.grad
    grad = h + scalar_lorentz(x, h) * x
    return grad


x, y = torch.randn(2, 4), torch.randn(2, 4)
print(x)
print(distance_lorentz(x, y))
