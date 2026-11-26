"""One full game plus a differentiable diffusion-path update check."""
import numpy as np
import torch
from tft_intense_game import BASE, game
from agent_measure.discrete_diffusion import CategoricalDenoiser
from agent_measure.diffusion_grpo import path_log_prob

def main():
    path=BASE/'tft_v2_suite_20261122/diffusion_seed247_beta0.pt'
    meta,rows=game({'seed':23001,'seat':1,'kind':'diffusion_grpo','collect':True,
                    'model_path':str(path),'sample_seed':77123})
    if meta['reason']!='natural_end' or not rows: raise RuntimeError(str(meta))
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=CategoricalDenoiser(checkpoint['obs_dim'],55*38)
    model.load_state_dict(checkpoint['model'])
    subset=rows[:8]
    x=torch.from_numpy(np.stack([r[0] for r in subset]))
    mask=torch.from_numpy(np.stack([r[1] for r in subset]))
    chains=torch.from_numpy(np.stack([r[3] for r in subset]))
    logp=path_log_prob(model,x,mask,chains)
    if not torch.isfinite(logp).all(): raise RuntimeError('nonfinite path logp')
    (-logp.mean()).backward()
    norm=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    if not np.isfinite(norm) or norm<=0: raise RuntimeError('invalid gradient')
    print('SMOKE_OK',meta,'records',len(rows),'gradient_norm',norm,flush=True)

if __name__=='__main__': main()
