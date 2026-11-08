"""2D counterpart of EXP-002: same 0.2 residual gate and shared dyadic map."""
import torch
from torch import nn
from .swe2d import carrier


class SpectralLayer(nn.Module):
    def __init__(self,width,modes):
        super().__init__()
        self.modes=modes
        self.weights=nn.Parameter(torch.randn(2,width,width,modes,modes,dtype=torch.cfloat)/width)

    def forward(self,u):
        f=torch.fft.rfft2(u)
        out=torch.zeros_like(f)
        m=min(self.modes,u.shape[-2]//2,f.shape[-1])
        out[:,:,:m,:m]=torch.einsum('bixy,ioxy->boxy',f[:,:,:m,:m],self.weights[0,:,:,:m,:m])
        out[:,:,-m:,:m]=torch.einsum('bixy,ioxy->boxy',f[:,:,-m:,:m],self.weights[1,:,:,:m,:m])
        return torch.fft.irfft2(out,s=u.shape[-2:])


class FlowMap(nn.Module):
    def __init__(self,kind,width=12,modes=4,depth=3,dt=.02):
        super().__init__()
        self.kind,self.dt=kind,dt
        self.lift=nn.Conv2d(3,width,1)
        self.spectral=nn.ModuleList([SpectralLayer(width,modes) for _ in range(depth)])
        self.pointwise=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(depth)])
        self.local=nn.ModuleList([nn.Conv2d(width,width,5,padding=2,padding_mode='circular') for _ in range(depth)])
        self.output=nn.Sequential(nn.Conv2d(width,width,1),nn.GELU(),nn.Conv2d(width,3,1))

    def forward(self,u):
        mean=u.mean((-2,-1),keepdim=True)
        features=torch.cat((u,(u+mean)/2),0) if self.kind=='dyadic' else u
        z=self.lift(features)
        for s,p,l in zip(self.spectral,self.pointwise,self.local):
            z=torch.nn.functional.gelu(s(z)+p(z)+l(z))
        neural=self.output(z)
        if self.kind=='dyadic':
            full,half=neural.chunk(2,0)
            neural=2*(full-2*half)
        inc=.2*neural
        base=carrier(u,self.dt) if self.kind in ('carrier','dyadic') else u
        return base+inc-inc.mean((-2,-1),keepdim=True)


def parameter_count(model):
    return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
