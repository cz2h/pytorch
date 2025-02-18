import torch

def empty_fn(size):
  return torch.empty(size)

opt_model = torch.compile(empty_fn)

print(opt_model((2, 3)))
print(opt_model((3, 4)))
# print(torch.empty((2,3)))