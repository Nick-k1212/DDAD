import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity

def get_loss(model, x_0, t, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)


    x = at.sqrt() * x_0 + (1- at).sqrt() * e 
    output = model(x, t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def get_lossSSIM(model, x_0, t, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)


    x = at.sqrt() * x_0 + (1- at).sqrt() * e 
    output = model(x, t.float())
    return structural_similarity(e, output, multichannel=True)

# .square(): 計算誤差的平方。
# .sum(dim=(1, 2, 3)): 對每個樣本的所有像素求和（計算單張圖的總誤差）。
# .mean(dim=0): 對 batch 求平均，得到最終的損失值。