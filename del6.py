import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import ast2000tools.constants as cs 
import tqdm
from ast2000tools.solar_system import SolarSystem
from numba import jit

seed = 59529
system = SolarSystem(seed)
np.random.seed(seed)

vmax_c = 1e4/cs.c

spectral_lines = {
    632: 'O2', 690: 'O2', 760: 'O2', 720: 'H2O',
    820: 'H2O', 940: 'H2O', 1400: 'CO2', 1600: 'CO2',
    1660: 'CH4', 2200: 'CH4', 2340: 'CO', 2870: 'N2O'
}

gasses = {
    'O2': {'name': 'Oxygen', 'A': 32}, 'H2O': {'name': 'Water vapor', 'A': 18},
    'CO2': {'name': 'Carbon dioxide', 'A': 44}, 'CH4': {'name': 'Methane', 'A': 16},
    'CO': {'name': 'Carbon monoxide', 'A': 28}, 'N2O': {'name': 'Nitrous dioxide', 'A': 44}
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
        for temp in range(150,451,5):
            for F_min in np.linspace(0.6, 1.0, 100):
                chisqr = chi_sqr(
                    flux[idx_lmb0-N:idx_lmb0+N], lmbda[idx_lmb0-N:idx_lmb0+N],
                    lmbda0, sigma[idx_lmb0-N:idx_lmb0+N], doppler, temp, F_min, m
                )
                chi_values.append(chisqr)
                param.append([doppler, temp, F_min])
    return chi_values, param



def atmosphere_chem_comp(lmbda, flux, sigma):
    
    print('Finner parametre ved chi^2-test:')
    parameters = []
    for lmbda0 in tqdm.tqdm(spectral_lines.keys()):
        gas = spectral_lines[lmbda0]
        gas_info = gasses[gas]
        m = gas_info['A']*cs.m_p
        i = np.argmin(abs(lmbda-lmbda0))
        chi_values, param = chi_squared_test(lmbda, lmbda0, i, flux, sigma, m)
        i_min = np.argmin(chi_values)
        
        parameters.append(param[i_min])

    return parameters



def plot_model_over_data(flux, lmbda, dlmbda, parameters):
    print('*'*90)
    print('| Gas                    | lambda0  [nm]  | dlambda [nm]  | Temperature [K]  | F_min     |')
    print('*'*90)

    j = 0
    fig, axs = plt.subplots(nrows=3, ncols=4)
    ymin, ymax = 0.5, 1.5
    for lmbda0 in spectral_lines.keys():
    
        i = np.argmin(abs(lmbda-lmbda0))
        N = int(lmbda0*vmax_c/dlmbda)

        doppler, temp, F_min = parameters[j]
        gas = spectral_lines[lmbda0]
        gas_info = gasses[gas]
        m = gas_info['A']*cs.m_p
        name = gas_info['name']
    
        print(f'| {name:<16} ({gas:<3}) | {lmbda0:<15.2f}| {doppler:<14.5f}| {temp:<17}| {F_min:<10.5f}|')
    
        std = (lmbda0/cs.c)*np.sqrt(cs.k_B*temp/m)
        F = 1 +(F_min-1)*np.exp(-((lmbda[i-N:i+N]+doppler-lmbda0)**2)/(2*std**2))
        axs[j//4, j%4].set_ylim(ymin, ymax)
        axs[j//4, j%4].plot(lmbda[i-N:i+N], flux[i-N:i+N])
        axs[j//4, j%4].plot(lmbda[i-N:i+N], F, color='red', label=f'{gas}: {lmbda0}nm')
        axs[j//4, j%4].legend(loc='upper right')

        j += 1
    
    print('*'*90)
    for ax in axs.flat:
        ax.label_outer()
    fig.supxlabel('Bølgelengde $\\lambda$ [nm]')
    fig.supylabel('Normalisert fluks $F$')
    plt.show()



def plot_sigma(sigma, lmbda, dlmbda):
    fig, axs = plt.subplots(nrows=3, ncols=4)
    ymin, ymax = min(sigma), max(sigma)
    j = 0
    for lmbda0 in spectral_lines.keys():
        gas = spectral_lines[lmbda0]
        i = np.argmin(abs(lmbda-lmbda0))
        N = int(lmbda0*vmax_c/dlmbda)
        
        axs[j//4, j%4].set_ylim(ymin, ymax)
        axs[j//4, j%4].plot(lmbda[i-N:i+N], sigma[i-N:i+N], label=f'$\\sigma_i$ rundt $\\lambda_0$ = {lmbda0}nm')
        axs[j//4, j%4].legend(loc='upper right')
                
        j += 1
    
    for ax in axs.flat:
        ax.label_outer()
    fig.supxlabel('Bølgelengde $\\lambda$ [nm]')
    fig.supylabel('Standard avvik for støyet $\\sigma_i$')
    plt.show()


def temperature(r):
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    T0 = 271
    rho0 = system.atmospheric_densities[1]
    r0 = system.radii[1]*1e3
    M_T = system.masses[1]*cs.m_sun
    gamma = 1.4

    frac = r0*T0*gamma*cs.k_B/(2*(gamma-1)*mu*cs.G*M_T)
    r_iso = r0 / (1 - frac)
    
    if r > r_iso:
        T = T0/2
    else:
        T = T0 - (gamma-1)/gamma * mu*cs.G*M_T/cs.k_B * (1/r0 - 1/r)
    
    return T

def pressure(r):
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    T0 = 271
    rho0 = system.atmospheric_densities[1]
    r0 = system.radii[1]*1e3
    p0 = rho0*cs.k_B*T0/mu
    M_T = system.masses[1]*cs.m_sun
    gamma = 1.4
    
    frac = r0*T0*gamma*cs.k_B/(2*(gamma-1)*mu*cs.G*M_T)
    r_iso = r0 / (1 - frac)
    
    if r > r_iso:
        p_iso = p0 * (T0/temperature(r_iso))**(gamma/(1-gamma))
        p = p_iso * np.exp(-(2*mu*cs.G*M_T)/cs.k_B/T0 * (1/r_iso - 1/r))
    else:
        p = p0 * (T0/temperature(r))**(gamma/(1-gamma))
    
    return p

def density(r):
    mu = (gasses['CO']['A']+gasses['CH4']['A'])*cs.m_p/2
    return pressure(r)*mu/cs.k_B/temperature(r)
    

lmbda, flux = np.load("spectrum_644nm_3000nm.npy")[:,0], np.load("spectrum_644nm_3000nm.npy")[:,1]
sigma = np.load("sigma_noise.npy")[:,1]
dlmbda = (lmbda[-1]-lmbda[0])/len(lmbda)

#parameters = atmosphere_chem_comp(lmbda, flux, sigma)
#plot_model_over_data(flux, lmbda, dlmbda, parameters)
# plot_sigma(sigma, lmbda, dlmbda)


r = np.linspace(system.radii[1]*1e3,system.radii[1]*1e3 + 1.3e6,10000)
temp, pres, dens = [], [], []
for ri in r:
    temp.append(temperature(ri))
    pres.append(pressure(ri))
    dens.append(density(ri))

fig, axs = plt.subplots(1,3)

axs[0].semilogy(r, temp)
axs[1].semilogy(r, pres)
axs[2].semilogy(r, dens)
plt.show()