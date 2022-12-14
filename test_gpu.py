import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Devices count:",torch.cuda.device_count())