#modules
import numpy as np 
import matplotlib.pyplot as plt 
import sys ,os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rv-pipeline'))
from doppler import doppler_C , doppler_r
from astropy import constants as const

#low velocities

wav = np.array([5000])
velocities = np.linspace(-100,100,1000)

shift_c = np.array([doppler_C(wav,v_t)[0] for v_t in velocities])
shift_r = np.array([doppler_r(wav,v_t)[0] for v_t in velocities])

c_ms = const.c.to("m/s").value
diff_ms = (shift_r-shift_c)/wav[0] *c_ms

#high velocities


high_v = np.linspace(0,0.9*const.c.to("km/s").value,1000)

shift_c_hv = np.array([doppler_C(wav,v_t)[0] for v_t in high_v])
shift_r_hv = np.array([doppler_r(wav,v_t)[0] for v_t in high_v])

#plots

fig, axes = plt.subplots(1,3,figsize=(12,12))

#plot 1 stable RV regime

ax = axes[0]
ax.plot(velocities,shift_c,'b--',linewidth = 2.5,label = 'classical')
ax.plot(velocities,shift_r,'r--',linewidth = 2.,label = 'Relativistic')
ax.set_xlabel('velocity')
ax.set_ylabel('shift wavelength')
ax.set_title('Classical and relativistic shift ')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)


ax2= axes[1]
ax2.plot(velocities,diff_ms,'r--',linewidth = 2.,label = 'Residuals')
ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
ax2.axhline(16.7, color='black', linestyle='--', linewidth=1.5,label='16.7 is Max absolute  residual for +- 100km/s velocities')
ax2.set_xlabel('velocities')
ax2.set_ylabel('residuals')
ax2.set_title('Residuals between classical and relativistic at stable rv ')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

ax3 =axes[2]
ax3.plot(high_v,shift_c_hv,'b--',linewidth = 2.5,label = 'classical')
ax3.plot(high_v,shift_r_hv,'r--',linewidth = 2.,label = 'Relativistic')

deltaV = ((np.abs(shift_r_hv-shift_c_hv))/wav)*const.c.to("km/s").value
v = deltaV# this is te difference between the shifted of both classical and relativistic

threshold = 5e3 #m/s
split_idk = np.where(deltaV>threshold)[0][0]# any value above threshold is accepted her and the first value is selected to find the velocity at which spilt happens
v_spilt = high_v[split_idk]# converting the residual back to velocity single value
ax3.axvline(v_spilt,color="black",linestyle="--",linewidth = 1.5,label = "threshold = 5e3 #m/s")


ax3.set_xlabel('high velocity')
ax3.set_ylabel('shift wavelength')
ax3.set_title('Classical and relativistic shift for high velocity')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('C:/Users/chand/OneDrive/Desktop/RV/rv-pipeline/results/day03_doppler_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# verifying the functions

v_t = 45
wav = 5000.0
expected_shift = wav*v_t/const.c.to("km/s").value
actual_shift =  doppler_C(np.array([wav]),v_t)[0] - wav

print(f"expected shift = {expected_shift}")
print(f"actual shift = {actual_shift}")
assert np.isclose(expected_shift,actual_shift,rtol = 1e-10),"shift formula wrong"
print("passed")

