"""Small controlled 2-D diffusion A/B diagnostic, not an RL benchmark."""
import json
import math
import time
import torch
from torch import nn

torch.set_num_threads(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T, N, STEPS, BATCH = 32, 8192, 2500, 256
beta_gibbs = 0.8
schedule = torch.linspace(1e-4, 0.08, T, device=device)
alpha = 1 - schedule
abar = alpha.cumprod(0)

def data(seed, n):
    g = torch.Generator().manual_seed(seed)
    mode = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    x = torch.randn(n, 2, generator=g) * 0.5
    x[:, 0] += 2 * mode
    return x.to(device)

def cost(x):
    return (x[:, 0] - 2).square() / 4 + x[:, 1].square() / 4

def mmd(x, y):
    # Fixed Gaussian kernel, fixed evaluation sample count and bandwidth.
    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b).square() / (2 * 2.0**2)).mean()
    return (kernel(x,x) + kernel(y,y) - 2*kernel(x,y)).item()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    def forward(self, x, t):
        return self.net(torch.cat((x, t.float().unsqueeze(1) / T), 1))

@torch.no_grad()
def sample(model, n):
    x = torch.randn(n, 2, device=device)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        mean = (x - (1-alpha[i]) / torch.sqrt(1-abar[i]) * model(x,t)) / torch.sqrt(alpha[i])
        if i:
            variance = schedule[i] * (1-abar[i-1]) / (1-abar[i])
            x = mean + torch.sqrt(variance) * torch.randn_like(x)
        else:
            x = mean
    return x

def run(seed):
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    x = data(31415, N)
    logits = -beta_gibbs * cost(x)
    weights = torch.softmax(logits, 0)
    ess = (1 / weights.square().sum()).item()
    # Evaluation target is independently sampled, then independently resampled.
    target_pool = data(27182, 20000)
    target_w = torch.softmax(-beta_gibbs * cost(target_pool), 0)
    targets = {}
    results = {}
    for group in ('A', 'B'):
        torch.manual_seed(seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
        model = Net().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        start = time.monotonic()
        for step in range(STEPS):
            idx = torch.randint(N, (BATCH,), device=device)
            batch = x[idx]
            t = torch.randint(T, (BATCH,), device=device)
            noise = torch.randn_like(batch)
            noisy = abar[t,None].sqrt()*batch + (1-abar[t,None]).sqrt()*noise
            per_item = (model(noisy,t)-noise).square().mean(1)
            loss = (per_item * (N * weights[idx] if group == 'B' else 1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        generated = sample(model, 2048)
        torch.manual_seed(98765)
        if device.type == 'cuda': torch.cuda.manual_seed_all(98765)
        target_indices = torch.multinomial(target_w if group == 'B' else torch.ones_like(target_w), 1024, replacement=True)
        target = target_pool[target_indices]
        results[group] = {'mean_cost': cost(generated).mean().item(), 'right_mode_fraction': (generated[:,0]>0).float().mean().item(),
                          'mmd_to_own_target': mmd(generated[:1024],target), 'seconds': elapsed}
        targets[group] = {'mean_cost': cost(target).mean().item(), 'right_mode_fraction': (target[:,0]>0).float().mean().item()}
    return {'seed':seed, 'ess_fraction': ess/N, 'targets':targets, 'groups':results}

if __name__ == '__main__':
    output = {'device':str(device), 'torch':torch.__version__, 'train_size':N, 'steps':STEPS, 'batch_size':BATCH,
              'diffusion_steps':T, 'gibbs_beta':beta_gibbs, 'runs':[run(i) for i in (11,22,33)]}
    with open('toy_results.json','w') as f: json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))
