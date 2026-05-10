import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rv-pipeline'))
from noise import add_noise
from doppler import doppler_shift_classical , doppler_shift_relativistic
from spectra import Gaussian_line


wav = np.linspace(4980,5030,5000)

#parameters 

wav0 = np.array([4985.0,4990.0,5000.0,5015.0,5025.0])
sigma = np.array([0.4,0.5,0.6,0.7,0.45])
depth = np.array([0.6,0.4,0.8,0.5,0.35])
amp = 1.0

#starting with just a continuum then adding all the absorption lines

flux_clean = np.ones_like(wav)

for i in range(len(wav0)):
    flux_clean -= depth[i]*Gaussian_line(wav,wav0[i],sigma[i],amp)


v_r = 35.0 # known value 
wav_shifted = doppler_shift_classical(wav,v_r)

# evaluate the flux_clean which is defined on the points of wav_shifted at the point of wav 
flux_shifted = np.interp(wav,wav_shifted,flux_clean)

snr= 100
flux_noisy = add_noise(flux_shifted,snr=snr,seed =42)


c_kms = const.c.to('km/s').value
# for all wavelength line_centers , the expected shift and the actual center shift 
for i,wv0 in enumerate(wav0):

    expected_shift = wv0 * v_r/c_kms

    actual_center_shifted = wv0 + expected_shift

    print(f" line {i+1} = {wv0:.4f}| expected shift = {expected_shift:.4f}| actual center shifted = {actual_center_shifted:.4f}")

    # only the wavelength where the wavelength - wavelength_new_center 


for i ,wv0 in enumerate(wav0):

    expected_shift = wv0 * v_r/c_kms

    actual_center_shifted = wv0 + expected_shift  


    mask = np.abs(wav - actual_center_shifted) < 2.0 
# jut to make sure tha no negative values being there
    if mask.sum() > 0:
        # finding the index of deepest point from the flux-shifted 
        min_flux = np.argmin(flux_shifted[mask])
        # filtering all the wavelength to wavelength  who has shifted less than 2 angstrom
        wav_window = wav[mask]
        # from the filter wavelength find the index of the deepest point 
        actual_min_wav = wav_window[min_flux]
        residual_ms = np.abs(actual_min_wav - actual_center_shifted) / actual_center_shifted * c_kms*1000
        print(f"actual min = {actual_min_wav:.4f} | residuals = {residual_ms:.4f}")


fig ,axes = plt.subplots(1,3,figsize= (18,8))

ax = axes[0]
ax.plot(wav,flux_clean,'k--',linewidth=1.2,label= "template (rest frame)",zorder = 3)
ax.plot(wav,flux_noisy,'-',color="blue",alpha = 0.6 , label = f"observed (v_r = {v_r} km/s, snr = {snr})")
ax.set_xlabel("wavelength")
ax.set_ylabel("normalized flux",fontsize = 11)
ax.set_title("synthetic spectrum pipeline")
ax.legend(fontsize = 10)
ax.grid(alpha =0.2)

#residual(noisy - clean) shows the structure of noise
ax2 = axes[1]
ax2.plot(wav,flux_noisy-flux_clean,linestyle = "--",linewidth = 1.2)
ax2.axhline(0,color= 'black',linestyle= "--",linewidth = 1.0)
ax2.axhline(+1.0/snr,color="red",linestyle = ":",linewidth = 1.0,label = "+1sigma")
ax2.axhline(-1.0/snr,color="red",linestyle = ":",linewidth = 1.0,label = "-1sigma")
ax.set_ylabel('residual(noise-clean)',fontsize =11)
ax2.legend(fontsize = 10)
ax2.grid(alpha = 0.3)

# fro a specific wav0 5000 angstrom
ax3 = axes[2]
z_mask = np.abs(wav-5000.0) < 3.0
ax3.plot(wav[z_mask],flux_clean[z_mask],'k-',linewidth=2,label = "template")
ax3.plot(wav[z_mask],flux_noisy[z_mask],'r-',linewidth=2,label = "noisy")
ax3.plot(wav[z_mask],flux_shifted[z_mask],'-',color = 'blue',linewidth=2,label = f"shifted  = {v_r}")

ax3.set_xlabel('wavelength',fontsize= 11)
ax3.set_ylabel("normalized flux ",fontsize = 11)
ax3.set_title("5000 angstrom line template vs shifted",fontsize = 11)
ax3.legend(fontsize =11)
ax3.grid(alpha = 0.3)

plt.tight_layout()
plt.savefig('C:/Users/chand/OneDrive/Desktop/RV/rv-pipeline/results/day07_integration_test.png',dpi=150, bbox_inches='tight')

plt.show()

# assert where there should be no flux lower than 0 and not higher than 1 plus some tolerance
assert np.all(flux_clean >= 0.0),"negative flux is there"
assert np.all(flux_clean <= 1.0+1e-10),"flux exceeds the continuum" 
print(f"flux upper and lower bound checked") 


# shift  direction should be correct as its v>0 then redshift as move to right 
# the 5000 angstrom line minimum should be at greater than 5000 wavelength
mask_5000 = wav-5000 < 2.0
wav_5_clean = wav[mask_5000][(np.argmin(flux_clean[mask_5000]))]
wav_5_shifted = wav[mask_5000][(np.argmin(flux_shifted[mask_5000]))]
assert wav_5_shifted > wav_5_clean , f"minimum moved left | wav_clean = {wav_5_clean:.4f} | wav_shifted = {wav_5_shifted:.4f}"
print(f"shift direction checked")

# shift magnitude check
expected_shift_5000 = 5000*v_r/c_kms
actual_shift_5000 = wav_5_shifted-wav_5_clean
assert np.abs(actual_shift_5000-expected_shift_5000) < 0.01,f"magnitude wrong | expected shift = {expected_shift_5000 :.4f} | actual shift at 5000 = {actual_shift_5000:.4f}"
print(f'magnitude check passed')

# SNR check at continuum 
# need this for calculating snr
flux_noisy_c = add_noise(flux_clean,snr=snr,seed =42)
continuum_mask = flux_clean > 0.95
measured_snr = 1 / np.std(flux_noisy_c[continuum_mask]- flux_clean[continuum_mask])
assert np.abs(measured_snr-snr)/snr < 0.15 ,f"SNR off , target snr = {snr} | measured_snr = { measured_snr:.1f}"
print(f"snr check done")


