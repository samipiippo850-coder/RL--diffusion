"""Check whether copied PettingZoo states replay identically under fixed actions.

This is a feasibility test, not proof that copying captures every simulator RNG.
"""
import copy
import hashlib
import json
import random
import time
import contextlib
import io
import numpy as np
from Simulator import TFTConfig,parallel_env
from tft_opponents import econ_policy,rank_events
from tft_pilot_train import features

def roll(bundle,steps):
    env,obs,infos,rngs,alive,ranks=bundle
    digest=hashlib.sha256();events=0;warnings=0
    for _ in range(steps):
        if not env.agents:break
        actions={}
        for a in env.agents:
            mask=np.asarray(obs[a]['action_mask'],dtype=bool).reshape(-1)
            action=econ_policy(mask,infos.get(a,{}),rngs[a],obs[a]);actions[a]=action
            digest.update(a.encode());digest.update(int(action).to_bytes(4,'little'))
            digest.update(np.asarray(mask,dtype=np.uint8).tobytes())
            digest.update(features(obs[a]).tobytes())
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            obs,rewards,terms,truncs,infos=env.step(actions)
        warnings+=sum(any(w in l.lower() for w in ('failed','could not','should never')) for l in captured.getvalue().splitlines())
        ended=[a for a,done in terms.items() if done and a in alive]
        if ended:
            ranked=rank_events(ended,alive,infos,rewards)
            if ranked is None:raise RuntimeError('ambiguous terminal')
            ranks.update(ranked);alive.difference_update(ended)
        for a in sorted(rewards):digest.update(np.float64(rewards[a]).tobytes())
        events+=1
    return ({'steps':events,'digest':digest.hexdigest(),'ranks':dict(ranks),'alive':sorted(alive),'warnings':warnings},
            (env,obs,infos,rngs,alive,ranks))

def main():
    started=time.monotonic();results=[]
    for seed in (61001,61002):
        random.seed(seed);np.random.seed(seed)
        env=parallel_env(TFTConfig(num_players=8));obs,infos=env.reset(seed=seed)
        rngs={a:np.random.default_rng(seed*100+i+8192) for i,a in enumerate(env.possible_agents)}
        bundle=(env,obs,infos,rngs,set(env.agents),{})
        _,bundle=roll(bundle,120)
        py_state=random.getstate();np_state=np.random.get_state()
        first=copy.deepcopy(bundle);second=copy.deepcopy(bundle)
        random.setstate(py_state);np.random.set_state(np_state);a,_=roll(first,900)
        random.setstate(py_state);np.random.set_state(np_state);b,_=roll(second,900)
        result={'seed':seed,'exact_match':a==b,'first':a,'second':b}
        results.append(result);print(json.dumps(result),flush=True)
        env.close();first[0].close();second[0].close()
    print('FEASIBILITY',json.dumps({'all_match':all(x['exact_match'] for x in results),
                                   'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)

if __name__=='__main__':main()
