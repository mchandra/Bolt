import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')

import pylab as pl 

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20  
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

h5f         = h5py.File('/home/mchandra/bolt/run_folder/dump.h5', 'r')
density     = h5f['density'][:]
vel_drift_x = h5f['vel_drift_x'][:]
vel_drift_y = h5f['vel_drift_y'][:]
T_ee        = h5f['T_ee'][:]
x_center = h5f['x_center'][:]
y_center = h5f['y_center'][:]
time     = h5f['time'][:]
h5f.close()

print("End time = ", time[-1])

#h5f         = h5py.File('/home/mchandra/bolt/run_folder/dump_defect_AC.h5', 'r')
#density_defect     = h5f['density'][:]
#vel_drift_x_defect = h5f['vel_drift_x'][:]
#vel_drift_y_defect = h5f['vel_drift_y'][:]
#h5f.close()

vel_mag = (vel_drift_x[:, 3:-3, 3:-3]**2. + vel_drift_y[:, 3:-3, 3:-3]**2.)**0.5

#min_val = np.min(vel_drift_x[:, :, :])
#max_val = np.max(vel_drift_x[:, :, :])
#colorlevels = np.linspace(min_val, max_val, 100)

#time_step = 1
#pl.figure(figsize=(15, 6.5))
#pl.contourf(x_center[3:-3, 3:-3, 0], y_center[3:-3, 3:-3, 0], vel_mag[time_step], colorlevels, cmap='gist_heat')
#pl.axes().set_aspect('equal')
#pl.colorbar()
#
#pl.savefig('vel_mag.png')

mean_density = np.mean(density[0, 3:-3, 3:-3])
density_pert = density[:, 3:-3, 3:-3]-mean_density
min_val = np.min(density_pert)
max_val = np.max(density_pert)
colorlevels = np.linspace(min_val, max_val, 100)

print("Mean density = ", mean_density)
#pl.plot(time, density_pert[:, 31, 15])
#pl.ylabel('Density')
#pl.xlabel('Time')
#pl.savefig('density_vs_time.png')

#for time_step in range((int)(time.size)-1, (int)(time.size)):
for time_step in range(0, (int)(time.size), 10):
    print("time_step = ", time_step)

    pl.contourf(x_center[3:-3, 3:-3, 0], y_center[3:-3, 3:-3, 0], density_pert[time_step], colorlevels, cmap='bwr')
    pl.colorbar()
    pl.streamplot(x_center[3:-3, 3:-3, 0], y_center[3:-3, 3:-3, 0], \
                  vel_drift_x[time_step, 3:-3, 3:-3], vel_drift_y[time_step, 3:-3, 3:-3], \
                  density=2, color='blue', linewidth=0.7, arrowsize=1)
                  #density=5, color='blue', linewidth=5*vel_mag[time_step]/vel_mag.max(), arrowsize=0.01)

    pl.xlim([0., 0.5])
    pl.ylim([0., 1.])
    pl.axes().set_aspect('equal')
    pl.title(r'Time = ' + "%.2f"%(0.01*time_step) )
    pl.savefig('device_sim_' + '%06d'%time_step + '.png')
    pl.clf()

#mean_density_defect = np.mean(density_defect[0, 3:-3, 3:-3])
#density_pert_defect = density_defect[:, 3:-3, 3:-3]-mean_density_defect
#min_val_defect = np.min(density_pert_defect)
#max_val_defect = np.max(density_pert_defect)
#colorlevels_defect = np.linspace(min_val_defect, max_val_defect, 100)
#
#for time_step in range(0, (int)(time.size), 10):
#    print("time_step = ", time_step)
#
#    pl.subplot(121)
#    pl.contourf(x_center[3:-3, 3:-3, 0], y_center[3:-3, 3:-3, 0], density_pert_defect[time_step], colorlevels_defect, cmap='bwr')
#    pl.streamplot(x_center[3:-3, 3:-3, 0], y_center[3:-3, 3:-3, 0], \
#                  vel_drift_x_defect[time_step, 3:-3, 3:-3], vel_drift_y_defect[time_step, 3:-3, 3:-3], \
#                  density=2, color='blue', linewidth=0.7, arrowsize=1)
#                  #density=5, color='blue', linewidth=5*vel_mag[time_step]/vel_mag.max(), arrowsize=0.01)
#
#    pl.xlim([0., 0.5])
#    pl.ylim([0., 1.])
#    pl.gca().set_aspect('equal')
#
#    pl.subplot(122)
#    pl.contourf(x_center[3:-3, 3:-3, 0], y_center[3:-3, 3:-3, 0], density_pert[time_step], colorlevels, cmap='bwr')
#    pl.streamplot(x_center[3:-3, 3:-3, 0], y_center[3:-3, 3:-3, 0], \
#                  vel_drift_x[time_step, 3:-3, 3:-3], vel_drift_y[time_step, 3:-3, 3:-3], \
#                  density=2, color='blue', linewidth=0.7, arrowsize=1)
#                  #density=5, color='blue', linewidth=5*vel_mag[time_step]/vel_mag.max(), arrowsize=0.01)
#
#    pl.xlim([0., 0.5])
#    pl.ylim([0., 1.])
#    pl.gca().set_aspect('equal')
#    pl.suptitle(r'Time = ' + "%.2f"%(0.01*time_step) )
#    #pl.colorbar()
#    pl.savefig('device_sim_' + '%06d'%time_step + '.png')
#    pl.clf()
#pl.semilogy(time, np.sum(vel_mag, axis=(1, 2)) )
#pl.savefig('vel_mag_vs_time.png')

#left_wall_minus_eps = vel_drift_x[:, 3:-3, 2]
#left_wall_plus_eps  = vel_drift_x[:, 3:-3, 3]
#
#right_wall_minus_eps = vel_drift_x[:, 3:-3, -3]
#right_wall_plus_eps  = vel_drift_x[:, 3:-3, -2]
#
#source_current = np.sum(left_wall_minus_eps  + left_wall_plus_eps,  axis=1)
#drain_current  = np.sum(right_wall_minus_eps + right_wall_plus_eps, axis=1)
#
#start_index = (int)(3*time.size/4)
#
#pl.plot(time[start_index:], source_current[start_index:], label=r'Source')
#pl.plot(time[start_index:], drain_current[start_index:], label=r'Drain')
#pl.legend()
#pl.xlabel(r'Time')
#pl.ylabel(r'Current')
#pl.savefig(r'source_vs_drain.png')



