import arrayfire as af
import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
import glob
import os
import sys
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import transforms, colors
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc
import PetscBinaryIO

import domain
#import boundary_conditions
#import params
#import initialize
#import coords


def line(x, m, c):
    return (m*x+c)


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'normal'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = False
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

# High res theta
theta_left   = np.loadtxt('theta_left.txt')
theta_right  = np.loadtxt('theta_right.txt')
theta_bottom = np.loadtxt('theta_bottom.txt')
theta_top    = np.loadtxt('theta_top.txt')

x = np.arange(theta_left.size + theta_top.size + theta_right.size + theta_bottom.size)

theta_left[:20] = theta_left[:20] + np.pi
theta_right[:20] = theta_right[:20] - np.pi
theta_bottom = theta_bottom - np.pi

theta = theta_left
theta = np.append(theta, theta_top)
theta = np.append(theta, np.flip(theta_right))
theta = np.append(theta, np.flip(theta_bottom))


pl.subplot(211)
ax1 = pl.gca()
pl.plot(x[:theta_left.size], theta_left, '-o')
pl.plot(x[theta_left.size:theta_left.size+theta_top.size], theta_top, '-o')
pl.plot(x[theta_left.size+theta_top.size:theta_left.size+theta_top.size+theta_right.size], np.flip(theta_right), '-o')
pl.plot(x[theta_left.size+theta_top.size+theta_right.size:], np.flip(theta_bottom), '-o')

#pl.axvline(x[21], color = 'k', alpha = 0.5)
#pl.axvline(x[109], color = 'k', alpha = 0.5)

#pl.axhline(np.pi, color = 'k', ls = '--')
#pl.axhline(-np.pi, color = 'k', ls = '--')
pl.axhline(np.pi/2, color = 'k', ls = '--')
pl.axhline(-np.pi/2, color = 'k', ls = '--')
pl.axhline(0, color = 'k', ls = '--')

popt, pcov = curve_fit(line, np.arange(theta.size), theta)
best_fit = line(np.arange(theta.size), popt[0], popt[1])

#pl.plot(np.arange(theta.size), best_fit, color = 'k', ls = '--')

#np.savetxt('theta_left_new.txt', best_fit[:40])
#np.savetxt('theta_top_new.txt', best_fit[40:80])
#np.savetxt('theta_right_new.txt', np.flip(best_fit[80:120]))
#np.savetxt('theta_bottom_new.txt', np.flip(best_fit[120:160]))

    
pl.ylabel(r'$\theta$')

pl.subplot(212)
ax2 = pl.gca()
pl.plot(np.arange(theta.size), (theta-line(np.arange(theta.size), popt[0], popt[1])), '-o')


#pl.plot(np.arange(theta.size), best_fit, color = 'k', ls = '--')

#np.savetxt('theta_left_new.txt', best_fit[:40])
#np.savetxt('theta_top_new.txt', best_fit[40:80])
#np.savetxt('theta_right_new.txt', np.flip(best_fit[80:120]))
#np.savetxt('theta_bottom_new.txt', np.flip(best_fit[120:160]))


#pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$')
pl.savefig('images/iv.png')
pl.clf()

