import torch

def tensor_to_numpy(tensor):
    return tensor.permute(1,2,0).numpy()

def numpy_to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float()   