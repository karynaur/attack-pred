from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from vicreg import VICRegNet
from torch.utils.data import DataLoader


class SwapDataset:
    def __init__(self, features, noise):
        self.features = features
        self.noise = noise
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        
        sample = self.features[idx, :].copy()
        sample_ = self.swap_sample(sample)
        
        return {
            'x' : torch.tensor(sample_, dtype=torch.float),
            'y' : torch.tensor(sample, dtype=torch.float)            
        }
    
    def swap_sample(self,sample):
            num_samples = self.features.shape[0]
            num_features = self.features.shape[1]
            if len(sample.shape) == 2:
                batch_size = sample.shape[0]
                random_row = np.random.randint(0, num_samples, size=batch_size)
                for i in range(batch_size):
                    random_col = np.random.rand(num_features) < self.noise
                    sample[i, random_col] = self.features[random_row[i], random_col]
            else:
                batch_size = 1  
                random_row = np.random.randint(0, num_samples, size=batch_size)
                random_col = np.random.rand(num_features) < self.noise
                sample[ random_col] = self.features[random_row, random_col]                
            return sample


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class Model(nn.Module):
    def __init__(self, 
                embedding, 
                mlp, 
                scaled_train_data, train_labels, scaled_test_data, test_labels,
                swap_noise = 0.15,
                sim_coeff = 25.0, 
                std_coeff = 15.0, 
                cov_coeff = 1.0,
                batch_size = 2048 * 5,
                include_anomaly = False,
                lr = 0.9
                ):
        super(Model, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VICRegNet(embedding, mlp).to(self.device)
        self.swap_noise = swap_noise

        self.sim_loss = nn.MSELoss()
        

        # coeffs
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.batch_size = batch_size
        self.include_anomaly = include_anomaly
        self.lr = lr
        self.optimizer = LARS(self.model.parameters(), lr=self.lr)
        self.scaled_train_data = scaled_train_data
        self.train_labels = train_labels
        self.scaled_test_data = scaled_test_data
        self.test_labels = test_labels

        self.optimizer = LARS(self.model.parameters(), self.lr)


    def std_loss(self, z_a, z_b):
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        return std_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cov_loss(self, z_a, z_b):
        N = z_a.shape[0]
        D = z_a.shape[1]
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        cov_loss = self.off_diagonal(cov_z_a).pow_(2).sum() / D + self.off_diagonal(cov_z_b).pow_(2).sum() / D
        return cov_loss
    
    def compute_loss(self, x):
        repr_a, repr_b = self.model(x['x'].to(self.device)), self.model(x['y'].to(self.device))
        _sim_loss = self.sim_loss(repr_a, repr_b)
        _std_loss = self.std_loss(repr_a, repr_b)
        _cov_loss = self.cov_loss(repr_a, repr_b)

        loss = self.sim_coeff * _sim_loss \
            + self.std_coeff * _std_loss \
            + self.cov_coeff * _cov_loss 

        return loss

    # function to take numpy array and return dataloader
    def get_dataloaders(self):
        bs = self.batch_size
        
        if not self.include_anomaly:
            xtr = SwapDataset(self.scaled_train_data.values[np.where(self.train_labels == 0)], self.swap_noise)
        else:
            xtr = SwapDataset(self.scaled_train_data.values, self.swap_noise)

        xts = SwapDataset(self.scaled_test_data.values, self.swap_noise)
        

        self.train_swap_dataloader = DataLoader(xtr, batch_size = bs)
        self.test_swap_dataloader = DataLoader(xts, batch_size = bs)

        self.train_dataloader = DataLoader(torch.FloatTensor(self.scaled_train_data.values),
                         batch_size = bs)
        self.test_dataloader = DataLoader(torch.FloatTensor(self.scaled_test_data.values),
                         batch_size = bs)
    
    def train(self, epochs):
        self.get_dataloaders()
        for epoch in tqdm(range(epochs),position=0, total = epochs):
            losses=[]
            
            for step, data in enumerate(self.train_swap_dataloader):
                self.optimizer.zero_grad()

                loss = self.compute_loss(data)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                
            val_dl = iter(self.test_swap_dataloader)
            val_scores = [self.compute_loss(next(val_dl)).item() for i in range(len(val_dl))]
            
            print(f'Epoch: {epoch}   Loss: {np.mean(losses):.4f}    Val_Loss: {np.mean(val_scores):.4f}')
        
    def compute_vectors(self):
        self.model.eval()
        trd = iter(self.train_dataloader)
        ted = iter(self.test_dataloader)
        train_len = len(self.train_dataloader)
        test_len = len(self.test_dataloader)
        with torch.no_grad():
            train_preds = np.vstack([self.model(next(trd).to(self.device)).cpu().data.numpy() for i in range(train_len)])
            test_preds = np.vstack([self.model(next(ted).to(self.device)).cpu().data.numpy() for i in range(test_len)])
        
        return train_preds, self.train_labels, test_preds, self.test_labels