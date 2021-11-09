import torch
from norm import two_norm

n,m = 8,3

a = torch.randn(n,m)
b = torch.randn(n,m)
c = torch.zeros(1)

print("a:\n",a)
print("\nb:\n",b)

a = a.cuda()
b = b.cuda()
c = c.cuda()

two_norm(b,a,c,n,m)

torch.cuda.synchronize()

print("\nresult by two_norm:",c)

print("\nresult by torch.norm:",torch.norm(a-b))
