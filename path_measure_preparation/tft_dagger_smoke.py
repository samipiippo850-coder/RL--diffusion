"""One real learner rollout with expert labels on the learner's states."""
from tft_intense_game import BASE,game
from tft_dagger_experiment import clean

def main():
    model=BASE/'tft_econ_imitation_20261124/hier_diffusion.pt'
    meta,rows=game({'seed':53001,'seat':1,'kind':'hier_diffusion','model_path':str(model),
                    'collect':True,'raw_collect':True,'dagger_collect':True})
    error=clean(meta,rows,True)
    if error:raise RuntimeError(error+' '+str(meta))
    if meta['dagger_count']!=len(rows):raise RuntimeError('label count mismatch')
    if meta['dagger_exact_agree']!=sum(int(r[2]==r[3]) for r in rows):raise RuntimeError('agreement count mismatch')
    print('DAGGER_SMOKE_OK',meta,'rows',len(rows),flush=True)

if __name__=='__main__':main()
