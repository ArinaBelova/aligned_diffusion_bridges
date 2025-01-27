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

from . optimal_weights import omega_optimized, gamma_by_gamma_max, gamma_by_r, gamma_by_range


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

def fbb(H, K=5, g_max=1.0, gamma_max=20.0,device="cpu"):
    print('init H in fbb',H)
    print('init K fbb',K)
    return FractionalSchrödingerBridge(H=H,K=K,g_max=g_max,gamma_max=gamma_max,device=device)

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
        self.K = 0
        self.H = 0.5
    def g(self, t):
        return np.ones_like(t) * self.g_max

class FractionalSchrödingerBridge(nn.Module):

    """Abstract class for an approximate fractional schrödinger bridge process"""

    def __init__(self, H=0.5, K=5, g_max=1.0, gamma_max=20.0, gamma_min=None, approx_cov=False, T=1.0, pd_eps=1e-4, device="cpu"):
        super(FractionalSchrödingerBridge, self).__init__()

        """parameters of fBM approximation"""
        self.register_buffer("H", torch.as_tensor(H, device=device))
        self.register_buffer("gamma_max", torch.as_tensor(gamma_max, device=device))
        if gamma_min is not None:
            self.register_buffer("gamma_min", torch.as_tensor(gamma_min, device=device))
        self.register_buffer("T", torch.as_tensor([[T]], device=device))
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
                if gamma_min is None:
                    gamma = gamma_by_gamma_max(K, self.gamma_max, device=device)
                else:
                    gamma = gamma_by_range(K, self.gamma_min, self.gamma_max)
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

        omega_i = self.omega[:,None,:].clone()
        omega_j = self.omega[:,:,None].clone()
        gamma_i = self.gamma[:,None,:].clone()
        gamma_j = self.gamma[:,:,None].clone()
        
        normalize_variance = False

        #only valid for T=1
        if normalize_variance:
            norm_constant = torch.sum((self.omega[:,None,:] * self.omega[:,:,None])/(self.gamma[:,None,:]+self.gamma[:,:,None]) * (1-torch.exp(-(self.gamma[:,None,:]+self.gamma[:,:,None])))).item()
        else:
            norm_constant=1.0
        
        print(f'normalize variance with {norm_constant}')
        self.g_max =  torch.tensor(g_max/norm_constant)
        print('dtype',self.g_max.dtype)
        print(f'g_max={g_max}')

        #self.g_max =  torch.tensor(g_max)

        if self.K>0:
            F = torch.zeros(K+1,K+1)
            F[:,1:] = -torch.vstack([self.g_max*(self.omega * self.gamma)[0],torch.diag(self.gamma[0])])
            self.register_buffer("F_t", F)

            G = torch.ones(K+1)
            G[0] = torch.sum(self.omega) * self.g_max
            self.register_buffer("G_t", G)

    def update_omega(self,omega,A=None,b=None):

        if A is not None:
            self.register_buffer("A", torch.as_tensor(A, device=self.device))
        if b is not None:
            self.register_buffer("b", torch.as_tensor(b, device=self.device))

        self.register_buffer("omega", torch.as_tensor(omega, device=self.device)[None, :].clone())
        self.register_buffer('sum_omega', torch.sum(self.omega))
        self.register_buffer("omega_i", self.omega[:, :, None].clone())
        self.register_buffer("omega_j", self.omega[:, None, :].clone())
        self.double_sum_omega = torch.sum(self.omega_i * self.omega_j, dim=(1, 2))

    def g(self,t):
        if self.K>0:
            return self.g_max
        else:
            return torch.ones_like(torch.tensor(t)) * self.g_max
    
    # def g(self,t):
    #     return self.g_max
    
    def zeta(self,s,t,gamma,g):

        # expects s,t of shape (batch_size1,batch_size2,1) and s<=t
        # expects omega and gamma of shape (1,1,K)

        return g*(torch.exp(-gamma*(t-s))-1)

    def meanX(self,s,t,x,Y,omega,gamma,g):

        # compute E[X(t)|Z_s=z) with s<t - Z_s = (x,Y) 
        # mean of X_T conditioned on Z_t = (x,Y)

        s = s[:,:,None]
        t = t[:,:,None]
        gamma = gamma[:,None,:]
        omega = omega[:,None,:]

        weight = omega * self.zeta(s,t,gamma,g) 
        y_part = (torch.sum(weight*Y, dim=-1)) 

        return x + y_part

    def meanY(self,s,t,gamma):

        # compute E[Y(t)|Z_s=z) with s<t 
        # mean of X_T conditioned on Z_t = (x,Y)

        s = s[:,:,None]
        t = t[:,:,None]
        gamma = gamma[:,None,:]

        return torch.exp(-gamma*(t-s)) * Y

    def meanZ(self,s,t,x,Y,omega,gamma,g):

        mean_x = self.meanX(s,t,x,Y,omega,gamma,g)
        mean_y = self.meanY(s,t,gamma)
    
        return torch.cat([mean_x.unsqueeze(-1),mean_y],dim=-1)
    
    def cond_var(self,t,T,omega,gamma,g):
                    
            # compute cov(X(t),X(t)|Z_s=z) with s<t 
            # expects s,t of shape (batch_size1,batch_size2,)
            # expects omega,gamma of shape (1,K)

            t = t[:,:,None,None] 
            T = T[:,:,None,None]

            return self.covX(t,T,T, omega, gamma, g)
    
    def covX(self,s,t,T, omega, gamma, g):

        # compute cov(X(t),X(T)|Z_s=z) with s<t<=T 
        # expects s,t,T of shape (batch_size1,batch_size2,1,1)
        # expects omega,gamma of shape (1,K)

        omega_ij = omega[:,None,:,None]*omega[:,None,None,:]
        gamma_i = gamma[:,None,:,None]
        gamma_j = gamma[:,None,None,:]
        gamma_ij =  gamma_i + gamma_j

        weight = omega_ij/ gamma_ij

        S = weight * (torch.exp(t*gamma_ij) -torch.exp(s*(gamma_ij))) * torch.exp(-T*gamma_j - t*gamma_i) 

        return g**2 * (torch.sum(S,axis=(2,3)))

    def covYX(self,s,t,T, omega, gamma, g):

        # compute cov(Y(t),X(T)|Z_s=z) with s<t<=T 
        # expects s,t,T of shape (batch_size1,batch_size2,1,1)
        # expects omega,gamma of shape (1,K)

        gamma_l = gamma[:,None,:,None] #dim of Y_l
        omega_k = omega[:,None,None,:]
        gamma_k = gamma[:,None,None,:]

        weight = omega_k/(gamma_l+gamma_k) #dim of omega_k in X
        S = weight *(torch.exp(t*(gamma_l+gamma_k)) -1)*torch.exp(-t*gamma_l-T*gamma_k)

        return g * torch.sum(S,axis=3)

    def covY(self,s,t,T, gamma):

        # compute cov(Y(t),Y(T)|Z_s=z) with s<t<=T 
        # expects s,t,T of shape (batch_size1,batch_size2,1,1)
        # expects omega,gamma of shape (1,K)

        gamma_i = gamma[:,None,:,None]
        gamma_j = gamma[:,None,None,:]
        gamma_ij =  gamma_i + gamma_j

        return (torch.exp(-T*gamma_j-t*gamma_i)*(torch.exp(t*gamma_ij)-torch.exp(s*gamma_ij)))/gamma_ij

    def covZ(self,t,T, omega, gamma, g, s=None, eps=1e-4):

        # compute cov(X(t),X(T)|Z_s=z) with s<t<=T 
        # expects s,t,T of shape (batch_size1,batch_size2,)
        # expects omega,gamma of shape (1,K)

        t = t[:,:,None,None] 
        T = T[:,:,None,None]
        s = torch.zeros_like(t) if s is None else s[:,:,None,None]

        K = omega.shape[1]
        bs1 = t.shape[0]
        bs2 = t.shape[1]
        Sig = torch.zeros(bs1, bs2, K+1,K+1)
        
        Sig_xy = self.covYX(s,t,T, omega, gamma, g)
        Sig[:,:,0,0] = self.covX(s,t,T, omega, gamma, g)
        Sig[:,:,1:,0] = Sig_xy
        Sig[:,:,0,1:] = Sig_xy
        Sig[:,:,1:,1:] = self.covY(s,t,T, gamma)

        # I_eps = torch.eye(K, K)[None, :, :] * torch.ones((bs, K, K)) * eps * torch.exp(-2 * gamma * t[:,:,0])[:, :, None]
        # Sig[:,1:,1:] = Sig[:,1:,1:] + I_eps
        # Sig[:,0,0] += eps

        Sig = Sig + torch.eye(K+1, K+1)[None,None, :, :] * torch.ones((bs1, bs2, K+1, K+1)) * eps

        assert ((torch.diag(Sig[0,0])>0).all()), f'Found negativ variance: \n {torch.diag(Sig[0,0])<0}'

        return Sig

    def sample_pinned(self,t,T,x0,xT,omega,gamma,g):

        K = omega.shape[1]
        print(f"IN DIFFUSIVITZ TIME SHAPE IS {t.shape}")

        bs1 = t.shape[0]
        bs2 = 1#t.shape[1]
        D = x0.shape[1]

        t = t[:,:,None,None] 
        T = T[:,:,None,None]
        s = torch.zeros_like(t)

        mu = torch.zeros(x0.shape+(K+1,))
        mu[:,:,0] = x0

        Sig_zx = torch.zeros(bs1,bs2,K+1)
        
        Sig_zx[:,:,0] = self.covX(s,t,T, omega, gamma, g)
        Sig_zx[:,:,1:] = self.covYX(s,t,T, omega, gamma, g)

        var = self.covX(s,T,T, omega, gamma, g)
        mu_bar = mu + (1/var[:,:,None]) * Sig_zx * ((xT-x0)[:,:,None])

        Sig_bar = self.covZ(t[:,:,0,0],t[:,:,0,0],omega,gamma,g) - (1/var[:,:,None,None]) * (Sig_zx[:,:,:,None] * Sig_zx[:,:,None,:])

        Sig_bar_flat = einops.rearrange(Sig_bar, 'bs1 bs2 K L -> (bs1 bs2) K L', bs1=bs1, bs2=bs2)
        assert (Sig_bar_flat.transpose(1, 2) == Sig_bar_flat).all(), f'Covariance is not symmetric'

        noise_flat = sample_from_batch_multivariate_normal(Sig_bar_flat,c=D,h=1,w=1,batch_size=int(bs1*bs2), aug_dim=K+1)[:,:,0,0,:]
        noise = einops.rearrange(noise_flat, '(bs1 bs2) D K -> bs1 bs2 D K', bs1=bs1, bs2=bs2)

        return mu_bar + noise

    def input_transform(self,x,Y,t,T,omega,gamma,g):

        t = t[:,:,None]
        T = T[:,:,None]
        gamma = gamma[:,None,:]
        omega = omega[:,None,:]

        weight = omega * self.zeta(t,T,gamma,g) 
        print(f"weight shape: {weight.shape}")
        print(f"Y shape is {Y.shape}")
        y_part = (torch.sum(weight*Y, dim=-1)) 

        return x + y_part

    def score(self, score_x, t,T, omega, gamma, g_max):

        # expects the output of a score model of dimension (batch_size1,batch_size2)

        omega = omega[:,None,:]
        gamma = gamma[:,None,:]

        scale = torch.ones(1,1,self.omega.shape[1]+1)
        scale[:,:,1:] = omega * self.zeta(t,T, gamma, g_max)
        
        return scale * score_x[:,:,None]
    
    # def mean_scale(self, t):
    #     return torch.exp(self.integral(t))

    # def mean(self,x0,t):
    #     c_t = self.mean_scale(t)[:,None,None,None,None]
    #     bs,c,h,w = x0.shape
    #     return torch.cat([(c_t*x0[:,:,:,:,None]),torch.zeros(bs,c,h,w,self.K,device=x0.device)],dim=-1)

    # def brown_moments(self,x0,t):
    #     return self.mean_scale(t)[:,None,None,None]*x0, torch.sqrt(self.brown_var(t))[:,None,None,None]

    # def augmented_var(self, t):
    #     return torch.diagonal(self.cov(t), dim1=1, dim2=2)

    # def forward_var(self, t):
    #     return self.augmented_var(t)[:, 0]

    # def f(self,z0,t):
    #     bs = t.shape[0]
    #     F_t = torch.cat([self.mu(t)[:,None],-self.gamma.repeat(bs,1)],dim=-1)[:,None,None,None,:]
    #     z1 = F_t * z0
    #     z1[:,:,:,:,0] = z1[:,:,:,:,0] + self.g(t)[:,None,None,None] * torch.sum(self.omega[:,None,None,None,:]*z1[:,:,:,:,1:],dim=-1)
    #     return z1

    # def G(self,t):
    #     M=1 if len(t.shape)==0 else t.shape[0]
    #     return torch.cat([(self.sum_omega * self.g(t))[:,None,None,None,None],torch.ones(M,self.K,device=t.device)[:,None,None,None,:]],dim=-1)

    # def prior_logp(self,z):
    #     if self.K==0:
    #         shape = z.shape
    #         N = np.prod(shape[1:])
    #         var_T = self.brown_var(self.T).detach().cpu().item()
    #         logp = -N / 2. * np.log(2 * np.pi * var_T) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2. * var_T)
    #     else:
    #         logp = self.terminal(z)
    #     return logp

    # def marginal_stats(self,t,batch=None):

    #     eps = self.pd_eps
    #     mean = self.mean(batch, t) if batch is not None else None
    #     cov = self.cov(t)
    #     bs = cov.shape[0]
    #     sigma_t = torch.squeeze(cov).clone().to(t.device)

    #     if bs==1:
    #         sigma_t = sigma_t[None, :, :]

    #     I_eps = torch.eye(self.aug_dim, self.aug_dim,device=t.device)[None, :, :] * torch.ones((t.shape[0],self.aug_dim, self.aug_dim),device=t.device)
    #     I_eps[:,1:,1:] = I_eps[:,1:,1:] * (eps * torch.exp(-2 * self.gamma * t[:,None])[:,:,None])
    #     I_eps[:, 0, 0] = 0.0
    #     sigma_t = sigma_t + I_eps

    #     corr = sigma_t[:,1:,0].clone()
    #     cov_yy = sigma_t[:,1:,1:].clone()
    #     var_x = sigma_t[:,0,0].clone()
    #     alpha = torch.linalg.solve(cov_yy,corr)
    #     var_c = torch.sum(alpha*corr,dim=-1)
    #     return sigma_t[:,None,None,None], mean, corr, cov_yy, alpha[:,None,None,None,:], var_x[:,None,None,None], var_c[:,None,None,None]

    # def compute_YiYj(self,t):
    #     sum_gamma = self.gamma_i + self.gamma_j
    #     return ((1-torch.exp(-t*sum_gamma))/sum_gamma)

    # def numpy_compute_YiYj(self,t):
    #     gamma_i, gamma_j = self.gamma[0,:, None].cpu().numpy(), self.gamma[0,None, :].cpu().numpy()
    #     return (1 - np.exp(- (gamma_i + gamma_j) * t.cpu().numpy())) / (gamma_i + gamma_j)

    # def func(self,t, S):
    #     num_k = self.K
    #     t = torch.as_tensor(t)
    #     A = np.zeros((num_k + 1, num_k + 1))
    #     A[0, 0] = 2 * self.mu(t).cpu().numpy()
    #     A[0, 1:] = - 2 * (self.g(t) * self.omega[0] * self.gamma[0]).cpu().numpy()
    #     A[1:, 1:] = np.diag((self.mu(t) - self.gamma[0]).cpu().numpy())
    #     b = np.zeros(num_k + 1)
    #     b[0] = (self.omega[0].cpu().numpy().sum() * self.g(t).cpu().numpy()) ** 2
    #     b[1:] = self.g(t).cpu().numpy() * (
    #                 self.omega[0].cpu().numpy().sum() - self.numpy_compute_YiYj(t) @ (self.omega[0] * self.gamma[0]).cpu().numpy())

    #     return A @ S + b

    # def solve_cov_ode(self,t=0.0):
    #     S_0 = np.zeros(self.K + 1)
    #     self.approx_sigma = solve_ivp(self.func, (t, 1.), S_0, dense_output=True)

#     @abstractmethod
#     def mu(self,t):
#         pass

#     @abstractmethod
#     def g(self,t):
#         pass

#     @abstractmethod
#     def integral(self,t):
#         pass

#     @abstractmethod
#     def brown_var(self,t):
#         pass

#     @abstractmethod
#     def compute_cov(self,t):
#         pass

#     @abstractmethod
#     def cov(self, t):
#         pass

#     @abstractmethod
#     def terminal(self, z):
#         pass """

# """ class ConstantFSB(FractionalDiffusion):

#     def __init__(self, H=0.5, K=5, g_max=1.0, gamma_max=20.0, gamma_min=None, T=1, pd_eps=0.0001, device="cpu"):
#         super().__init__(H=H, gamma_max=gamma_max, gamma_min=gamma_min, approx_cov=False, K=K, T=T, pd_eps=pd_eps, device=device)

#         self.g_max = g_max
#         self.solve_cov_ode()

#     def mu(self,t):
#         return torch.zeros_like(t)

#     def g(self, t):
#         t = torch.tensor(t)
#         return torch.ones_like(t) * self.g_max

#     def integral(self,t):
#         return torch.zeros_like(t)

#     def brown_var(self,t):
#         return t

#     def compute_cov(self,t):

#         S = self.approx_sigma.sol(t[:,0,0].cpu().numpy())
#         cov = np.zeros((self.K + 1, self.K + 1,t.shape[0]))
#         cov[0, :, :] = S
#         cov[:, 0, :] = S
#         sigma_t = torch.from_numpy(cov.astype(np.float32)).to(t.device).permute(2,1,0)
#         sigma_t[:,1:,1:] = self.compute_YiYj(t)
#         return sigma_t[:,None,None,None,:,:]

#     def cov(self, t):
#         if len(t.shape)==0:
#             t = t[None,None,None]
#         elif  len(t.shape)==1:
#             t =  t[:,None,None]
#         elif len(t.shape)==2:
#             t =  t[:,:,None]
#         return self.compute_cov(t)

#     def terminal(self, z):
#         pass
    
#     def sample(self, t, c=2, h=1, w=1):
#         batch_size = t.shape[0]
#         cov_matrix, _, _, _, _, _, _ = self.marginal_stats(t[:,0], batch=torch.zeros(batch_size,c,h,w))
#         if h==1 and w==1:
#             sample = sample_from_batch_multivariate_normal(torch.squeeze(cov_matrix), c=c, h=h, w=w,
#                                                                batch_size=batch_size, aug_dim=self.K+1)[:,:,0,0]
#         else:
#             sample = sample_from_batch_multivariate_normal(torch.squeeze(cov_matrix), c=c, h=h, w=w,
#                                                                batch_size=batch_size, aug_dim=self.K+1)
#         return sample

#     def pinned_statistics(self,t,a,b):
#         #print('t',t.shape)
#         #print('self.T',self.T.shape)
#         ktT = self.transition_kernel(t,torch.ones_like(t)*self.T)

#         #print(f"Gammas are {self.gamma}")
#         #print(f"Transition kernel shape of ktT is {ktT.shape}")
#         k0T = self.transition_kernel(torch.zeros_like(t),torch.ones_like(t)*self.T)
#         #print(f"Transition kernel shape of k0T is {k0T.shape}")
#         inv_k0T = torch.linalg.inv(k0T)
#         #print(f"Transition kernel shape of inv_k0T is {inv_k0T.shape}")
#         #print(f"Transition kernel shape of  TRANSPOSE of inv_k0T is {inv_k0T.mT.shape}")

#         k0t = self.transition_kernel(torch.zeros_like(t),t)
#         #print(f"Transition kernel shape of k0t is {k0t.shape}")
#         #print(f"Transition kernel shape of TRANSPOSE k0t is {k0t.mT.shape}")

#         #print('a',a.shape)
#         #print('b',b.shape)

#         mean = ((ktT @ inv_k0T).unsqueeze(1) @ a.unsqueeze(-1) + (k0t.mT @ inv_k0T.mT).unsqueeze(1) @ b.unsqueeze(-1)).squeeze(-1)
#         #print('mean',mean.shape)
#         cov = ktT @ inv_k0T @ k0t
#         eps=1e-4
#         #eps = 1.0
#         #print('epsilson for covariance matrix:',eps)
        
#         # I_eps = torch.eye(self.aug_dim, self.aug_dim,device=t.device)[None, :, :] * torch.ones((t.shape[0],self.aug_dim, self.aug_dim),device=t.device)
#         # I_eps[:,0,0] = I_eps[:,0,0] * eps
#         # I_eps[:,1:,1:] = I_eps[:,1:,1:] * (eps * torch.exp(-2 * self.gamma * t[:,None])[:,:,None])
#         I_eps = torch.eye(self.aug_dim, self.aug_dim,device=t.device) * eps

#         #I_eps = I_eps * (eps * torch.exp(-2 * self.gamma * t[:,None])[:,:,None])
#         #I_eps[:,1:,1:] = I_eps[:,1:,1:] * eps 
        
#         cov = cov + I_eps

#         #print(f"Eigenvelus of covariance matrix: {np.linalg.eigvals(cov)}")
#         #print('cov',cov)

#         # cov = cov[0]
#         # mean = mean[0]
#         return mean, cov
    
    
#     def transition_kernel(self,s,t):
#         eps = torch.mean(self.gamma) * 1e-4 #1e-1 #this 1e-1 is too large, but choosing it smaller results in non positive definit covariance matrix
#         lam = torch.cat([torch.tensor([-eps]),-self.gamma[0]])
        
#         #print(f'eigenvalues for times from {s} to {t}: {lam}')
#         lam_ij = lam[None,:] + lam[:,None]
        
#         # print("time shape ", t.shape)
#         # print("G shape ", self.G(t[:,0])[:,0,0,0,:])
#         G = self.G(t[:,0])[:,0,0,0,:]
#         return ((1/(lam_ij)) * (torch.exp(lam_ij*t[:,:,None])-torch.exp(lam_ij*s[:,:,None]))) * (G[:,:,None]*G[:,None,:])
    
#     def pinned_marginals(self,t,a,b):
#         mean,cov = self.pinned_statistics(t,a,b)
#         return mean + sample_from_batch_multivariate_normal(cov, c=2,h=1,w=1,batch_size=t.shape[0], aug_dim=self.K+1).squeeze()


# class SDE(ABC, nn.Module):
#     def __init__(self, device="cpu",D=1):
        
#         self.D=D
    
#     def func_cov(self,t, S):
#         t = torch.as_tensor(t)
#         return self.F(t) @ S + S @ self.F(t).T + self.G(t) @ self.G(t).T
    
#     def func_mean(self,t, m):
#         t = torch.as_tensor(t)
#         return self.F(t) @ m + self.u(t) 

#     def solve_cov_ode(self,t=0.0):
#         S_0 = np.zeros(self.K + 1)
#         self.approx_sigma = solve_ivp(self.func, (t, 1.), S_0, dense_output=True)

#     @abstractmethod
#     def drift(self,t):
#         pass
    
#     @abstractmethod
#     def G(self,t):
#         pass

#     @abstractmethod
#     def u(self,t):
#         pass
    
# class PinnedSDE():

#     def __init__(self, D):
#         super().__init__(dif,zT,D=D)

#         self.dif = dif
#         self.zT = zT
    
#     def drift(self,t):
#         COV_Tt = torch.zeros(bs,K+1,K+1)
#         for i,s in enumerate(t):
#             COV_Tt[i] = self.dif.solve_ode(s)
#             inv_COV_Tt = self.invert(COV_Tt)
        
#         A = self.dif.F(t) + (self.G(t) @ self.G(t).T) @ self.expF(t).T @ inv_COV_Tt @ self.expF(t)
#         u = -dif.G(t) @ dif.G(t).T @ self.expF(t).T @ inv_COV_Tt @ self.zT
#         return A, u
    
#     def G(self,t):
#         return self.dif.G(t)
    
#     def expF(self,t):
#         return
    
#     def invert(self,S):
#         return S



def sample_from_batch_multivariate_normal(cov_matrix, c=2,h=1,w=1,batch_size=128, aug_dim=6, eps=1e-5,device='cpu'):

    # Ensure covariance matrix has shape [batch_size, dim, dim]
    assert cov_matrix.shape == (batch_size, aug_dim, aug_dim), "Covariance matrix must have shape [batch_size, dim, dim]"
    
    # Zero mean for each distribution in the batch
    mean = torch.zeros(batch_size, aug_dim,device=device)

    # Create the batch of Multivariate Normal distributions
    mvn = MultivariateNormal(mean, covariance_matrix=cov_matrix)

    # Sample from the distribution
    n_samples = int(c*h*w)
    samples = mvn.sample(sample_shape=(n_samples,))  # Samples will have shape [n_samples, batch_size, dim]
    samples = einops.rearrange(samples, '(C H W) B K -> B C H W K', C=c, H=h, W=w)

    return samples


def matrix_vector_mp(A,v):

    '''
    # Example tensors A and v
    A = torch.randn(128, 2, 1, 1, 6, 6)
    v = torch.randn(128, 2, 1, 1, 6)
    '''

    # Reshape v to have an additional dimension at the end (for matrix-vector multiplication)
    v_expanded = v.unsqueeze(-1)  # Now v has shape (128, 2, 1, 1, 6, 1)

    # Perform matrix-vector multiplication
    result = A @ v_expanded

    # The result will have shape (128, 2, 1, 1, 6, 1), you might want to remove the last dimension
    result_squeezed = result.squeeze(-1)  # Now result has shape (128, 2, 1, 1, 6)
    return result_squeezed


def fractional_data_transform(data, diffusivity_schedule, max_diffusivity, H=0.5, K=0):
    dif=get_diffusivity_schedule(diffusivity_schedule, max_diffusivity, H=H, K=K)
    if dif.K>0:
        data.pos_t, data.cond_var_t = fractional_input_transform(data.pos_t, data.t, data.aug_pos_T[:,:,1:], dif)
    #    _, _, _, _, eta_Tt, sig_Tt, tau_Tt = dif.marginal_stats(1.0 - data.t[:,0])
    #    data.cond_var_t =  (sig_Tt - tau_Tt)[:,0,0]
    #    varphi = dif.g(data.t) * dif.omega * dif.gamma * (1.0 - data.t) + torch.exp(-dif.gamma * (1.0 - data.t))
    #    data.pos_t = data.pos_t[:,:,0] + torch.sum(eta_Tt[:,0,0] * data.aug_pos_T[:,:,1:] + varphi[:,None,:] * data.pos_t[:,:,1:], dim=-1)
        
def fractional_input_transform(z_t, t, yT, dif):
    _, _, _, _, eta_Tt, sig_Tt, tau_Tt = dif.marginal_stats(1.0 - t[:,0])

    # print("eta_Tt ", eta_Tt.shape)
    # print("sig_Tt ", sig_Tt.shape)
    # print("tau_Tt ", tau_Tt.shape)

    cond_var_t =  (sig_Tt - tau_Tt)[:,0,0,:]
    varphi = torch.exp(-dif.gamma * (1.0 - t))
    return z_t[:,:,0] + torch.sum(eta_Tt[:,0,0,:,:] * yT - varphi[:,None,:] * z_t[:,:,1:], dim=-1), cond_var_t