import torch.nn as nn

def Projector(embedding, mlp):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

class VICRegNet(nn.Module):
    def __init__(self, embedding, mlp):
        super().__init__()
        self.expander = Projector(embedding, mlp)
        
    def forward(self, x):
        _embeds = self.expander(x)
        return _embeds
    
    def device(self):
        return next(self.parameters()).device.type
