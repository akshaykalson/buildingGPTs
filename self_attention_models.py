import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram_train import BigramLanguageModel

# Set a manual seed for reproducibility
torch.manual_seed(42)

# Create a lower triangular matrix and normalize it
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)

# Create a random matrix
b = torch.randint(0, 10, (3, 2)).float()

# Perform matrix multiplication
c = a @ b

# Print results
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

#self attention bock
# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x

print("Output shape:", out.shape)