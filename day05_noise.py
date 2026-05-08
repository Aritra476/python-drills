import numpy as np 
import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rv-pipeline'))
from spectra import Gaussian_line
from noise import add_noise


wav = np.linspace(4990,5010,1000)
wav0 = 5000.0 
sigma = 0.5
amp = 1.0
depth = 0.7

flux_clean = 1- depth*(Gaussian_line(wav,wav0,sigma,amp))
mask = flux_clean > 0.8

noisy_spectra = {}
snr_levels = [10,50,100,300]
for snr in snr_levels:
    noisy_spectra[snr] = add_noise(flux_clean,snr,seed =None)

fig , axes = plt.subplots(1,4,figsize =(18,8))
axes =axes.ravel() # convert to 1D
color = ['steelblue', 'darkorange', 'forestgreen', 'crimson']


for i ,snr in enumerate(snr_levels):

    ax = axes[i]
    noisy = noisy_spectra[snr]

    measure_std = np.std(noisy[mask] - flux_clean[mask])
    measure_snr = 1/measure_std

    ax.plot(wav,flux_clean,'k--',linewidth = 1.0,label ='clean',zorder= 3)
    ax.scatter(wav,noisy,s =2 ,color= color[i],alpha = 0.5,label =f'noisy SNR = {snr}',zorder = 2)
    ax.fill_between(wav,flux_clean-1.0/snr,flux_clean+1.0/snr,alpha = 0.2 , color=color[i],label = "± sigma band")
    ax.set_xlabel('wavelength',fontsize = 12)
    ax.set_ylabel("normalized flux",fontsize = 12)
    ax.set_title(f"Snr = {snr} | measured snr = {measure_snr} ",fontsize = 12)
    ax.legend(fontsize = 12,markerscale = 4)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.2,1.3)


plt.suptitle('Noise model: Poisson + read noise at four SNR levels', fontsize=13)
plt.tight_layout()
plt.savefig('C:/Users/chand/OneDrive/Desktop/RV/rv-pipeline/results/day05_noise_comparison.png',dpi=150, bbox_inches='tight')
plt.show()

print(f"SNR verification ")

for snr in snr_levels:
    noisy = noisy_spectra[snr]
    measure_std = np.std(noisy[mask] - flux_clean[mask])
    measure_snr = 1/measure_std
    assert abs(measure_snr - snr)  / snr < 0.1 , "SNR off by greater than 10 percentage"
print("Saved")