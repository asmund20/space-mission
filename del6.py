import numpy as np
import matplotlib.pyplot as plt









#data = np.genfromtxt("spectrum_seed29_600nm_3000nm.txt")#, delimiter="\t")
#noise = np.genfromtxt("sigma_noise.txt")#,delimiter="\t")
#np.save("spectrum_644nm_3000nm.npy", data)
#np.save("sigma_noise.npy", noise)
data = np.load("spectrum_644nm_3000nm.npy")
noise = np.load("sigma_noise.npy")
print(data.shape, noise.shape)
print(data[:15], noise[:15])
a = data+noise
plt.scatter(data[:,0], a[:,1])
plt.show()
