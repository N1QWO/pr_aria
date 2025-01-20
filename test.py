import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



outp_w = nn.Parameter(torch.randn(2))
f = F.softmax(outp_w,dim=0)

i = torch.rand(1,10)
p = torch.rand(1,10)

o = f[0] * i + f[1] * p
print('f',f)
print('i',i)
print('p',p)
print('o',o)