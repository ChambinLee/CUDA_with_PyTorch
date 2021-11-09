import torch
from torch.utils.cpp_extension import load

norm = load(name="two_norm",
                   sources=["two_norm/two_norm_bind.cpp", "two_norm/two_norm_kernel.cu"],
                   verbose=True)
n,m = 8,3

a = torch.randn(n,m)
b = torch.randn(n,m)
c = torch.zeros(1)

print("a:\n",a)
print("\nb:\n",b)

a = a.cuda()
b = b.cuda()
c = c.cuda()

norm.two_norm(a,b,c,n,m)

torch.cuda.synchronize()

print("\nresult by two_norm:",c)

print("\nresult by torch.norm:",torch.norm(a-b))
