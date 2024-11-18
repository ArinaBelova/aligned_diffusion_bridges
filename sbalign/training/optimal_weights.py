import torch
import torch.special as ts


def omega_optimized(gamma, hurst, time_horizon, return_cost=False, return_Ab = False,device='cpu'):
    # print(f'omega_optimized hurst={hurst}')
    """
    Based on mean of approximation error with type II fractional brownian motion.

    Args:
        gamma: quadrature values of the speed of mean reversion
        hurst: hurst-index
        time_horizon: time-horizon / largest time-step that is considered
        return_cost: of specified, the cost of the approximation is returned as well
        fbm_type: type of fractional brownian motion that is beeing dealt with (choose from {1, 2})
    Return:
        Approximation of the quadrature values, needed to approximate fractional brownian motion in markovian setting.
    """

    gamma = torch.as_tensor(gamma, device=device)
    time_horizon = torch.as_tensor(time_horizon, device=device)
  #  print('gamma',gamma)

    gamma_i, gamma_j = gamma[None, :], gamma[:, None]

    A = (time_horizon + (torch.exp(- (gamma_i + gamma_j) * time_horizon) - 1) / (gamma_i + gamma_j)) / (
                gamma_i + gamma_j)
    b = time_horizon / gamma ** (hurst + .5) * ts.gammainc(hurst + .5, gamma * time_horizon) - (
                hurst + .5) / gamma ** (hurst + 1.5) * ts.gammainc(hurst + 1.5, gamma * time_horizon)

    # solve the linear programm
    omega = torch.linalg.solve(A, b)

    #  print('A',A.shape)
    #  print('b', b.shape)
    #  print('omega', omega.shape)
    output = omega if not return_Ab else (omega,A,b)
    # return the cost if needed
    if return_cost:
        c = time_horizon ** (2 * hurst + 1) / (2 * hurst) / (2 * hurst + 1) / torch.exp(torch.lgamma(hurst + .5)) ** 2
        cost = 1 - b @ omega / c
        return output, cost
    else:
        return output

def gamma_by_r(num_k, r, offset=0.,device='cpu'):
    n = (num_k + 1) / 2 + offset
    k = torch.arange(1, num_k + 1,device=device)
    gamma = r ** (k - n)
    return gamma

def gamma_by_gamma_max(num_k, gamma_max, offset=0.,device='cpu'):
    r = gamma_max ** (2 / (num_k - 1 - 2 * offset))
    return gamma_by_r(num_k, r, offset,device=device)

def gamma_by_range(num_k, gamma_min, gamma_max):
    return torch.exp(torch.linspace(torch.log(gamma_min), torch.log(gamma_max), num_k))