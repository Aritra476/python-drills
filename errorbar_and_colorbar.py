import numpy as np 
import matplotlib.pyplot as plt

# reproducible random numbers
rng  = np.random.default_rng(seed = 42) 

x = np.linspace(0,10,100)
y = np.sin(x)
y_err = 0.1*np.ones_like(x)
y_noise = y + rng.normal(0,0.1,size =x.shape)

fig , axes = plt.subplots(1,2, figsize = (10,10) )

ax = axes[0]
ax.errorbar(x[::10],y_noise[::10],yerr=y_err[::10],fmt="ro", capsize = 4, label = 'measured')
ax.fill_between(x,y-y_err,y+y_err,alpha = 0.2 , color="blue")
ax.set_xlabel('x',fontsize = 10)
ax.set_ylabel('y',fontsize = 10)
ax.set_title("sin wave with error",fontsize = 10)
ax.legend(fontsize = 10)
ax.grid(alpha=0.2)

ax2 = axes[1]
sc = ax2.scatter(x[::10],y_noise[::10],c = np.abs(y_noise[::10] - y[::10]),cmap = "viridis", s = 20)
cb = plt.colorbar(sc)
cb.set_label("true y",fontsize = 10)
ax2.set_xlabel("x",fontsize = 10)
ax2.set_title("scatter with color bar",fontsize = 10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('drill_day05.png',dpi =140,bbox_inches='tight')
plt.show()



