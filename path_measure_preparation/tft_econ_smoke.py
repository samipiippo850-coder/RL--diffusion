"""Check raw demonstration capture and both factorized policies in real games."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from tft_intense_game import BASE,game
from tft_econ_imitation import inspect_episode
from agent_measure.hierarchical_policy import HierarchicalPolicy,HierarchicalDenoiser

def main():
    torch.set_num_threads(1)
    old=torch.load(BASE/'tft_pilot_20261121/policy.pt',map_location='cpu',weights_only=False)
    mean,std=old['mean'],old['std'];out=BASE/'tft_econ_smoke_20261124';out.mkdir(exist_ok=True)
    for kind,model in [('hier_bc',HierarchicalPolicy(len(mean))),('hier_diffusion',HierarchicalDenoiser(len(mean)))]:
        torch.save({'model':model.state_dict(),'mean':mean,'std':std},out/(kind+'.pt'))
    jobs=[{'seed':45101,'seat':5,'kind':'econ','collect':True,'raw_collect':True}]
    jobs += [{'seed':45101,'seat':5,'kind':k,'model_path':str(out/(k+'.pt'))} for k in ('hier_bc','hier_diffusion')]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(game,jobs))
    for job,(meta,rows) in zip(jobs,results):
        if meta['reason']!='natural_end':raise RuntimeError(str(meta))
        if job['kind']=='econ' and inspect_episode(meta,rows):raise RuntimeError('invalid raw samples')
        print('SMOKE',job['kind'],meta,'samples',len(rows),flush=True)
    print('SMOKE_OK',flush=True)

if __name__=='__main__':main()
