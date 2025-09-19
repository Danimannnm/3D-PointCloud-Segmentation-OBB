import torch
print(torch.cuda.is_available())   # True
print(torch.version.cuda)          # '12.4'
print(torch.cuda.get_device_name(0))