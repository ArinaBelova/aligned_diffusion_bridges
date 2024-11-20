import numpy as np
from functools import partial

import torch
import numpy as np
from scipy.integrate import solve_ivp
from torch.distributions import MultivariateNormal
import time

from abc import ABC, abstractmethod
import torch.nn as nn
import einops

from . optimal_weights import omega_optimized, gamma_by_gamma_max, gamma_by_r


   

def constant_g(g_max):
    return ConstantDiffusivitySchedule(g_max)
    #return np.ones_like(t) * g_max

def triangular_g(t, g_max):
    g_min = 0.85
    return g_max - 2 * np.abs(t - .5) * (g_max-g_min)

def inverse_triangular_g(t, g_max):
    g_min = .01
    return g_min - 2 * np.abs(t - .5) * (g_min-g_max)

def decreasing_g(t, g_max):
    g_min = .1
    return g_max - np.square(t) * (g_max-g_min)

def fbb(H, K=5, gamma_max=20.0, device="cpu"):
    return FBB(H=H,K=K,gamma_max=gamma_max,device=device)

diffusivity_schedules = {
    "constant": constant_g,
    "triangular": triangular_g,
    "inverse_triangular": inverse_triangular_g,
    "decreasing": decreasing_g,
    "fbb": fbb,
}

def get_diffusivity_schedule(schedule, g_max, H=0.5, K=5):
    if schedule.lower() == 'fbb':
        return diffusivity_schedules[schedule](H=H, K=K, g_max=g_max)
    else: 
        return diffusivity_schedules[schedule](g_max)
    #return partial(diffusivity_schedules[schedule], g_max=g_max)

class ConstantDiffusivitySchedule():
    def __init__(self, g_max):
        self.g_max = g_max
    def g(self, t):
        return np.ones_like(t) * self.g_max

class FractionalDiffusion(ABC, nn.Module):
    """Abstract class for all fractional diffusion dynamics of the forward process"""

    def __init__(self, H=0.5, K=5, gamma_max=20.0, approx_cov=False, T=1.0, pd_eps=1e-4, device="cpu"):
        super(FractionalDiffusion, self).__init__()

        """parameters of fBM approximation"""
        self.register_buffer("H", torch.as_tensor(H, device=device))
        self.register_buffer("gamma_max", torch.as_tensor(gamma_max, device=device))
        self.register_buffer("T", torch.as_tensor([T], device=device))
        self.K = K

        """parameters of augmented process"""
        self.aug_dim = K + 1
        self.pd_eps = pd_eps

        self.approx_cov = approx_cov
        self.device = device

        if self.K > 0:
            if self.K == 1:
                gamma = gamma_by_r(K, torch.sqrt(torch.tensor(gamma_max)), device=device)
            else:
                gamma = gamma_by_gamma_max(K, self.gamma_max, device=device)
            omega, A, b = omega_optimized(
                gamma, self.H, self.T, return_Ab=True, device=device
            )

        else:
            gamma = torch.tensor([0.0])
            omega = torch.tensor([1.0])
            A = torch.tensor([1.0])
            b = torch.tensor([1.0])

        self.register_buffer("gamma", torch.as_tensor(gamma, device=device)[None, :])
        self.register_buffer("gamma_i", self.gamma[:, :, None].clone())
        self.register_buffer("gamma_j", self.gamma[:, None, :].clone())
        self.update_omega(omega,A=A,b=b)

    def update_omega(self,omega,A=None,b=None):

        if A is not None:
            self.register_buffer("A", torch.as_tensor(A, device=self.device))
            #self.A = torch.as_tensor(A, device=self.device)
        if b is not None:
            self.register_buffer("b", torch.as_tensor(b, device=self.device))
            #self.b = torch.as_tensor(b, device=self.device)

        self.register_buffer("omega", torch.as_tensor(omega, device=self.device)[None, :].clone())
        self.register_buffer('sum_omega', torch.sum(self.omega))
        self.register_buffer("omega_i", self.omega[:, :, None].clone())
        self.register_buffer("omega_j", self.omega[:, None, :].clone())
        self.double_sum_omega = torch.sum(self.omega_i * self.omega_j, dim=(1, 2))

    def mean_scale(self, t):
        return torch.exp(self.integral(t))

    def mean(self,x0,t):
        c_t = self.mean_scale(t)[:,None,None,None,None]
        bs,c,h,w = x0.shape
        return torch.cat([(c_t*x0[:,:,:,:,None]),torch.zeros(bs,c,h,w,self.K,device=x0.device)],dim=-1)

    def brown_moments(self,x0,t):
        return self.mean_scale(t)[:,None,None,None]*x0, torch.sqrt(self.brown_var(t))[:,None,None,None]

    def augmented_var(self, t):
        return torch.diagonal(self.cov(t), dim1=1, dim2=2)

    def forward_var(self, t):
        return self.augmented_var(t)[:, 0]

    def f(self,z0,t):
        bs = t.shape[0]
        F_t = torch.cat([self.mu(t)[:,None],-self.gamma.repeat(bs,1)],dim=-1)[:,None,None,None,:]
        z1 = F_t * z0
        z1[:,:,:,:,0] = z1[:,:,:,:,0] + self.g(t)[:,None,None,None] * torch.sum(self.omega[:,None,None,None,:]*z1[:,:,:,:,1:],dim=-1)
        return z1

    def G(self,t):
        M=1 if len(t.shape)==0 else t.shape[0]
        return torch.cat([(self.sum_omega * self.g(t))[:,None,None,None,None],torch.ones(M,self.K,device=t.device)[:,None,None,None,:]],dim=-1)

    def prior_logp(self,z):
        if self.K==0:
            shape = z.shape
            N = np.prod(shape[1:])
            var_T = self.brown_var(self.T).detach().cpu().item()
            logp = -N / 2. * np.log(2 * np.pi * var_T) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2. * var_T)
        else:
            logp = self.terminal(z)
        return logp

    def marginal_stats(self,t,batch=None):

        eps = self.pd_eps
        mean = self.mean(batch, t) if batch is not None else None
        cov = self.cov(t)
        bs = cov.shape[0]
        sigma_t = torch.squeeze(cov).clone().to(t.device)

        if bs==1:
            sigma_t = sigma_t[None, :, :]

        I_eps = torch.eye(self.aug_dim, self.aug_dim,device=t.device)[None, :, :] * torch.ones((t.shape[0],self.aug_dim, self.aug_dim),device=t.device)
        I_eps[:,1:,1:] = I_eps[:,1:,1:] * (eps * torch.exp(-2 * self.gamma * t[:,None])[:,:,None])
        I_eps[:, 0, 0] = 0.0
        sigma_t = sigma_t + I_eps

        corr = sigma_t[:,1:,0].clone()
        cov_yy = sigma_t[:,1:,1:].clone()
        var_x = sigma_t[:,0,0].clone()
        alpha = torch.linalg.solve(cov_yy,corr)
        var_c = torch.sum(alpha*corr,dim=-1)
        return sigma_t[:,None,None,None], mean, corr, cov_yy, alpha[:,None,None,None,:], var_x[:,None,None,None], var_c[:,None,None,None]

    def compute_YiYj(self,t):
        sum_gamma = self.gamma_i + self.gamma_j
        return ((1-torch.exp(-t*sum_gamma))/sum_gamma)

    def numpy_compute_YiYj(self,t):
        gamma_i, gamma_j = self.gamma[0,:, None].cpu().numpy(), self.gamma[0,None, :].cpu().numpy()
        return (1 - np.exp(- (gamma_i + gamma_j) * t.cpu().numpy())) / (gamma_i + gamma_j)

    def func(self,t, S):
        num_k = self.K
        t = torch.as_tensor(t)
        A = np.zeros((num_k + 1, num_k + 1))
        A[0, 0] = 2 * self.mu(t).cpu().numpy()
        A[0, 1:] = - 2 * (self.g(t) * self.omega[0] * self.gamma[0]).cpu().numpy()
        A[1:, 1:] = np.diag((self.mu(t) - self.gamma[0]).cpu().numpy())
        b = np.zeros(num_k + 1)
        b[0] = (self.omega[0].cpu().numpy().sum() * self.g(t).cpu().numpy()) ** 2
        b[1:] = self.g(t).cpu().numpy() * (
                    self.omega[0].cpu().numpy().sum() - self.numpy_compute_YiYj(t) @ (self.omega[0] * self.gamma[0]).cpu().numpy())

        return A @ S + b

    def solve_cov_ode(self,t=0.0):
        S_0 = np.zeros(self.K + 1)
        self.approx_sigma = solve_ivp(self.func, (t, 1.), S_0, dense_output=True)

    @abstractmethod
    def mu(self,t):
        pass

    @abstractmethod
    def g(self,t):
        pass

    @abstractmethod
    def integral(self,t):
        pass

    @abstractmethod
    def brown_var(self,t):
        pass

    @abstractmethod
    def compute_cov(self,t):
        pass

    @abstractmethod
    def cov(self, t):
        pass

    @abstractmethod
    def terminal(self, z):
        pass

class FBB(FractionalDiffusion):

    def __init__(self, H=0.5, K=5, g_max=1.0, gamma_max=20, T=1, pd_eps=0.0001, device="cpu"):
        super().__init__(H=H, gamma_max=gamma_max, approx_cov=False, K=K, T=T, pd_eps=pd_eps, device=device)

        self.g_max = g_max

    def mu(self,t):
        return torch.zeros_like(t)

    def g(self, t):
        return torch.ones_like(t) * self.g_max

    def integral(self,t):
        return torch.zeros_like(t)

    def brown_var(self,t):
        return t

    def compute_cov(self,t):

        S = self.approx_sigma.sol(t[:,0,0].cpu().numpy())
        cov = np.zeros((self.K + 1, self.K + 1,t.shape[0]))
        cov[0, :, :] = S
        cov[:, 0, :] = S
        sigma_t = torch.from_numpy(cov.astype(np.float32)).to(t.device).permute(2,1,0)
        sigma_t[:,1:,1:] = self.compute_YiYj(t)
        return sigma_t[:,None,None,None,:,:]

    def cov(self, t):
        t= t[None,None,None] if len(t.shape)==0 else t[:,None,None]
        return self.compute_cov(t)

    def terminal(self, z):
        pass

    def pinned_statistics(self,t,a,b):
        ktT = self.transition_kernel(t,self.T)

        k0T = self.transition_kernel(0,self.T)
        inv_k0T = torch.linalg.inv(k0T)

        k0t = self.transition_kernel(0,t)

        mean = ktT @ inv_k0T @ a + k0t.mT @ inv_k0T.mT @ b

        cov = ktT @ inv_k0T @ k0t
        eps=1e-4
        I_eps = torch.eye(self.aug_dim, self.aug_dim,device=t.device)[None, :, :] * torch.ones((t.shape[0],self.aug_dim, self.aug_dim),device=t.device)
       # I_eps[:,1:,1:] = I_eps[:,1:,1:] * (eps * torch.exp(-2 * self.gamma * t[:,None])[:,:,None])
        I_eps[:,1:,1:] = I_eps[:,1:,1:] * eps 
        cov = cov + I_eps
        print('cov',cov)
        return mean, cov
    
    
    def transition_kernel(self,s,t):
        eps = torch.mean(self.gamma) * 1e-1
        lam = torch.cat([torch.tensor([-eps]),-self.gamma[0]])
        lam_ij = lam[None,:] + lam[:,None]
        G = self.G(t)[:,0,0,0,:]
        return ((1/(lam_ij)) * (torch.exp(lam_ij*t)-torch.exp(lam_ij*s))) * (G[:,:,None]*G[:,None,:])

class SDE(ABC, nn.Module):
    def __init__(self, device="cpu",D=1):
        
        self.D=D
    
    def func_cov(self,t, S):
        t = torch.as_tensor(t)
        return self.F(t) @ S + S @ self.F(t).T + self.G(t) @ self.G(t).T
    
    def func_mean(self,t, m):
        t = torch.as_tensor(t)
        return self.F(t) @ m + self.u(t) 

    def solve_cov_ode(self,t=0.0):
        S_0 = np.zeros(self.K + 1)
        self.approx_sigma = solve_ivp(self.func, (t, 1.), S_0, dense_output=True)

    @abstractmethod
    def drift(self,t):
        pass
    
    @abstractmethod
    def G(self,t):
        pass

    @abstractmethod
    def u(self,t):
        pass
    
class PinnedSDE():

    def __init__(self, D):
        super().__init__(dif,zT,D=D)

        self.dif = dif
        self.zT = zT
    
    def drift(self,t):
        COV_Tt = torch.zeros(bs,K+1,K+1)
        for i,s in enumerate(t):
            COV_Tt[i] = self.dif.solve_ode(s)
            inv_COV_Tt = self.invert(COV_Tt)
        
        A = self.dif.F(t) + (self.G(t) @ self.G(t).T) @ self.expF(t).T @ inv_COV_Tt @ self.expF(t)
        u = -dif.G(t) @ dif.G(t).T @ self.expF(t).T @ inv_COV_Tt @ self.zT
        return A, u
    
    def G(self,t):
        return self.dif.G(t)
    
    def expF(self,t):
        return
    
    def invert(self,S):
        return S
