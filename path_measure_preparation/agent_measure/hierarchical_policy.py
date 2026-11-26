"""Seven masked action families for the simulator's 55 x 38 action grid."""
import torch
from torch import nn
from agent_measure.discrete_diffusion import CategoricalDenoiser

FAMILY_NAMES=('move','sell','item','buy','pass','level','refresh')

def action_families(device=None):
    a=torch.arange(55*38,device=device)
    row,col=a//38,a%38
    family=torch.zeros_like(a)
    family[(row<=36)&(col==37)]=1
    family[(row>=37)&(row<=46)]=2
    family[(row>=47)&(row<=51)]=3
    family[row==52]=4;family[row==53]=5;family[row==54]=6
    return family

def factorized_log_probs(family_logits,action_logits,mask):
    """log P(family|s) + log P(action|family,s), both legality masked."""
    mask=mask.bool()
    if not mask.any(dim=1).all(): raise ValueError('empty action mask')
    family=action_families(mask.device)
    valid=torch.stack([(mask & (family==f)).any(dim=1) for f in range(7)],dim=1)
    log_family=torch.log_softmax(family_logits.masked_fill(~valid,-1e9),dim=1)
    output=torch.full_like(action_logits,-1e9)
    for f in range(7):
        allowed=mask & (family==f)
        # Rows without this family will be overwritten with the illegal value.
        conditional=torch.log_softmax(action_logits.masked_fill(~allowed,-1e9),dim=1)
        output=torch.where(allowed,log_family[:,f,None]+conditional,output)
    return output

class HierarchicalPolicy(nn.Module):
    def __init__(self,obs_dim,hidden=256):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(obs_dim,hidden),nn.ReLU())
        self.family=nn.Linear(hidden,7);self.actions=nn.Linear(hidden,55*38)
    def forward(self,obs,mask):
        h=self.body(obs)
        return factorized_log_probs(self.family(h),self.actions(h),mask)

class HierarchicalDenoiser(CategoricalDenoiser):
    def __init__(self,obs_dim,n_actions=55*38,steps=4,alpha=.1,hidden=256):
        super().__init__(obs_dim,n_actions,steps,alpha,hidden)
        self.family_head=nn.Linear(hidden,7)
    def forward(self,obs,xt,t,mask):
        z=torch.cat((obs,self.action_embedding(xt),self.time_embedding(t)),dim=-1)
        hidden=self.net[1](self.net[0](z))
        return factorized_log_probs(self.family_head(hidden),self.net[2](hidden),mask)
