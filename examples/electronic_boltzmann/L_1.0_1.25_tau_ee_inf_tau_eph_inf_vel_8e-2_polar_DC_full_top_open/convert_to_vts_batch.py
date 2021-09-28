#import sys,petsc4py
#petsc4py.init(sys.argv)
#from petsc4py import PETSc
import PetscBinaryIO
import os
import glob
import numpy as np
import scipy.io

#import domain
   
io = PetscBinaryIO.PetscBinaryIO()

N_q1 = 40#domain.N_q1
N_q2 = 50#domain.N_q2


filepath = os.getcwd()# + '/backup'
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))


start_index = 0

for file_number, dump_file in enumerate(moment_files[:]):

    print("File number = ", file_number, " of ", moment_files.size)
    dump_file = 'coords.bin'
    coords = io.readBinaryFile(dump_file)
    coords = coords[0].reshape(N_q2, N_q1, 21)
    
    x = coords[:, :, 0]; y = coords[:, :, 1]
    
    
    moments = io.readBinaryFile(moment_files[start_index+file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    lagrange_multipliers = \
        io.readBinaryFile(lagrange_multiplier_files[start_index+file_number])
    lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 5)
    
    density = moments[:, :, 0]
    vel_drift_x  = lagrange_multipliers[:, :, 3]
    vel_drift_y  = lagrange_multipliers[:, :, 4]
    
    matfile = "images/dump_%06d"%(file_number+start_index)
    scipy.io.savemat(matfile, mdict={'x': x.T, 'y': y.T, 'density': density.T, 'vel_drift_x': vel_drift_x.T, 'vel_drift_y': vel_drift_y.T})
