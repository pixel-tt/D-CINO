"""Grid-independent 2D function parameters and conservative master quadrature."""
import numpy as np
import torch
from .models.two_dimensional.swe2d import conservative, galilean, mean_velocity, restrict
WAVES=[(1,0),(0,1),(1,1),(1,-1),(2,1),(1,2)]

def parameters(count,seed):
    """Grid-independent function identities; draw parameters without observing fields."""
    rng=np.random.default_rng(seed); result=[]
    for i in range(count):
        p=dict(id=f'{seed}:{i}',index=i,hbar=float(rng.uniform(.9,1.2)))
        if i%2==0:
            p.update(family='smooth',waves=[dict(k=[kx,ky],phase=rng.uniform(0,2*np.pi,3).tolist(),
                amplitude=(.065*rng.uniform(-1,1,3)/np.sqrt(kx*kx+ky*ky)).tolist()) for kx,ky in WAVES])
        else:
            p.update(family='dam',center=rng.uniform(.3,.7,2).tolist(),radius=float(rng.uniform(.12,.22)),
                jump=float(rng.uniform(.25,.6)),speed=rng.uniform(-.12,.12,2).tolist())
        result.append(p)
    return result


def initial(items,n,amplitude=1.,master=512,device='cuda'):
    """Use one master-grid conservative quadrature for all reference/model grids.

    Positivity holds before averaging. Mean velocity is removed on the same
    master grid. Thus restricting 256/512 states to 64 gives identical initial
    cell averages, including at circular interfaces, to floating point error.
    """
    x=(np.arange(master)+.5)/master; x,y=np.meshgrid(x,x,indexing='ij'); result=[]
    for p in items:
        q=np.zeros((3,master,master)); q[0]=p['hbar']
        if p['family']=='smooth':
            for w in p['waves']:
                kx,ky=w['k']
                q+=amplitude*np.array(w['amplitude'])[:,None,None]*np.sin(2*np.pi*(kx*x+ky*y)[None]+np.array(w['phase'])[:,None,None])
        else:
            cx,cy=p['center']
            mask=np.minimum(abs(x-cx),1-abs(x-cx))**2+np.minimum(abs(y-cy),1-abs(y-cy))**2<p['radius']**2
            q[0]+=amplitude*p['jump']*(mask-mask.mean())
            q[1:]=amplitude*np.array(p['speed'])[:,None,None]*(2*mask-1)
        u=conservative(torch.tensor(q[None],dtype=torch.float64,device=device))
        u=galilean(u,-mean_velocity(u))
        result.append(restrict(u,n))
    return torch.cat(result)
