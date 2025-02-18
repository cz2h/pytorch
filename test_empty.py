import torch

def empty_fn(size, out):
  return torch.empty(size, out=out)

opt_model = torch.compile(empty_fn)

out1 = torch.empty(2, 3)
opt_model((2, 3), out1)
print(out1)

out2 = torch.empty(3, 4)
opt_model((3, 4), out2)
# opt_model((2, 3), out2)
print(out2)