import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import ast2000tools.constants as cs 
from numba import jit

np.random.seed(59529)

vmax_c = 1e4/cs.c

spectral_lines = {
    632: 'O2', 690: 'O2', 760: 'O2', 720: 'H2O',
    820: 'H2O', 940: 'H2O', 1400: 'CO2', 1600: 'CO2',
    1660: 'CH4', 2200: 'CH4', 2340: 'CO', 2870: 'N2O'
}

gasses = {
    'O2': {'name': 'Oxygen', 'Z': 16, 'N':16}, 'H2O': {'name': 'Water vapor', 'Z': 10, 'N':8},
    'CO2': {'name': 'Carbon dioxide', 'Z': 22, 'N':22}, 'CH4': {'name': 'Methane', 'Z': 10, 'N':6},
    'CO': {'name': 'Carbon monoxide', 'Z': 14, 'N':14}, 'N2O': {'name': 'Nitrous dioxide', 'Z': 22, 'N':22}
}



@jit(nopython=True)
def chi_sqr(f, lmbda, lmbda0, sigma, doppler, temp, F_min, m):
    std = (lmbda0/cs.c)*np.sqrt(cs.k_B*temp/m)
    F = 1 + (F_min-1)*np.exp(-((lmbda+doppler-lmbda0)**2)/(2*std**2))
    
    df = len(F)
    chi_squared = np.sum(((f-F)/sigma)**2)
    
    return chi_squared



@jit(nopython=True)
def chi_squared_test(lmbda, lmbda0, idx_lmb0, flux, sigma, m):

    N = int(lmbda0*vmax_c/dlmbda)

    chi_values = []
    param = []
    for doppler in np.linspace(-vmax_c*lmbda0, vmax_c*lmbda0,100):
        for temp in range(150,451,10):
            for F_min in np.linspace(0.6, 1.0, 100):
                chisqr = chi_sqr(
                    flux[idx_lmb0-N:idx_lmb0+N], lmbda[idx_lmb0-N:idx_lmb0+N],
                    lmbda0, sigma[idx_lmb0-N:idx_lmb0+N], doppler, temp, F_min, m
                )
                chi_values.append(chisqr)
                param.append([doppler, temp, F_min])
    return chi_values, param



def atmosphere_chem_comp(lmbda, lmbda0, idx_lmb0, flux, sigma, m):
    chi_values, param = chi_squared_test(lmbda, lmbda0, idx_lmb0, flux, sigma, m)
    i_min = np.argmin(chi_values)
    parameters = param[i_min]

    return parameters



lmbda, flux = np.load("spectrum_644nm_3000nm.npy")[:,0], np.load("spectrum_644nm_3000nm.npy")[:,1]
sigma = np.load("sigma_noise.npy")[:,1]
dlmbda = (lmbda[-1]-lmbda[0])/len(lmbda)

parameters = []
for lmbda0 in spectral_lines.keys():
    gas = spectral_lines[lmbda0]
    gas_info = gasses[gas]
    m = gas_info['Z']*cs.m_p + gas_info['N']*cs.m_p
    i = np.argmin(abs(lmbda-lmbda0))
    p = atmosphere_chem_comp(lmbda, lmbda0, i, flux, sigma, m)
    parameters.append(p)

print('*'*66)
print('| Gas                    | dlambda   | Temperature   | F_min     |')
print('*'*66)

j = 0
for lmbda0 in spectral_lines.keys():
    
    i = np.argmin(abs(lmbda-lmbda0))
    N = int(lmbda0*vmax_c/dlmbda)

    doppler, temp, F_min = parameters[j]
    gas = spectral_lines[lmbda0]
    gas_info = gasses[gas]
    m = gas_info['Z']*cs.m_p + gas_info['N']*cs.m_p
    name = gas_info['name']
    
    print(f'| {name:<16} ({gas:<3}) | {doppler:<10.5f}| {temp:<14}| {F_min:<10.5f}|')
    
    std = (lmbda0/cs.c)*np.sqrt(cs.k_B*temp/m)
    F = 1 +(F_min-1)*np.exp(-((lmbda[i-N:i+N]+doppler-lmbda0)**2)/(2*std**2))
    plt.plot(lmbda[i-N:i+N], flux[i-N:i+N])
    plt.plot(lmbda[i-N:i+N], F, color='red')

    plt.show()
    j += 1
print('*'*66)

    
