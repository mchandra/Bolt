import pylab as pl
import numpy as np

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 25
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

time_1 = np.loadtxt("time.txt")
signal_1 = np.loadtxt("V_LU.txt")

time_2 = np.loadtxt("../ballistic_rectifier/time.txt")
signal_2 = np.loadtxt("../ballistic_rectifier/V_LU.txt")



pl.plot(time_1[:-2000], signal_1[:-2000], label = "Right Injection")
pl.plot(time_2, signal_2, label = "Left Injection")
pl.xlabel("Time (ps)")
pl.ylabel("V$_{LU}$")
#pl.xlim(xmax = 700)

pl.legend(loc="best")

pl.savefig('images/iv.png')
pl.clf()    

