import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import ast2000tools.constants as cs 


np.random.seed(59529)

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


def chi_sqr(f, lmbda, lmbda0, sigma, df):
    T = 271
    gas = spectral_lines[lmbda0]
    gas_info = gasses[gas]
    m = gas_info['Z']*cs.m_p + gas_info['N']*cs.m_p

    std = (lmbda0/cs.c)*np.sqrt(cs.k_B*T/m)
    F = 1 + (0.7-1)*np.exp(-((lmbda-lmbda0)**2)/(2*std**2))
    plt.plot(lmbda, F)
    plt.show()
    chi_squared = np.sum(((f-F)/sigma)**2)
    p = chi2.cdf(chi_squared, df)
    return p

lmbda, flux = np.load("spectrum_644nm_3000nm.npy")[:,0], np.load("spectrum_644nm_3000nm.npy")[:,1]
sigma = np.load("sigma_noise.npy")[:,1]
noise = np.random.normal(scale=sigma, size=len(sigma))
df = len(lmbda)

for lmbda0 in spectral_lines.keys():
    p = chi_sqr(flux+noise, lmbda, lmbda0, sigma, df)
    print(f'{spectral_lines[lmbda0]}: {p}')

