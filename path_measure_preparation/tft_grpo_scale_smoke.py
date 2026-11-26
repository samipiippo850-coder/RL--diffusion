"""Exercise concurrent same-seed rollouts and a CUDA GRPO update end-to-end."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from tft_intense_game import BASE,game
from agent_measure.discrete_diffusion import CategoricalDenoiser
import tft_grpo_experiment as pilot

def main():
    path=BASE/'tft_v2_suite_20261122/diffusion_seed247_beta0.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    seed=33001
    jobs=[{'seed':seed,'seat':seed%8,'kind':'diffusion_grpo','collect':True,
           'model_path':str(path),'sample_seed':440000+j} for j in range(4)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        episodes=list(pool.map(game,jobs))
    if not all(pilot.valid(m,r) for m,r in episodes): raise RuntimeError('invalid smoke rollout')
    model=CategoricalDenoiser(ck['obs_dim'],55*38);model.load_state_dict(ck['model'])
    opt=torch.optim.Adam(model.parameters(),lr=1e-5)
    before={k:v.clone() for k,v in model.state_dict().items()}
    out=pilot.update(model,opt,episodes,lambda m,r:(8-m['rank'])/7,
                     torch.device('cuda' if torch.cuda.is_available() else 'cpu'),550)
    if not all(np.isfinite(x) for x in out['losses']):raise RuntimeError('nonfinite loss')
    diff=sum((v-before[k]).abs().sum().item() for k,v in model.state_dict().items())
    print('SMOKE_OK',{'ranks':[m['rank'] for m,_ in episodes], 'parameter_change':diff,
                     'update':out},flush=True)

if __name__=='__main__':main()
