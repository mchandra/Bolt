import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import PetscBinaryIO

import domain

N_q1 = domain.N_q1
N_q2 = domain.N_q2

scalar_da = PETSc.DMDA().create([N_q1, N_q2],
                                dof = 1,
                                stencil_width = 0
                               )

scalar_da.setUniformCoordinates(0,1,0,1)

density_vec   = scalar_da.createGlobalVec()
density_array = density_vec.getArray()

vel_drift_x_vec   = scalar_da.createGlobalVec()
vel_drift_x_array = vel_drift_x_vec.getArray()

vel_drift_y_vec   = scalar_da.createGlobalVec()
vel_drift_y_array = vel_drift_y_vec.getArray()

PETSc.Object.setName(density_vec,     'density'    )
PETSc.Object.setName(vel_drift_x_vec, 'vel_drift_x')
PETSc.Object.setName(vel_drift_y_vec, 'vel_drift_y')

vector_da = PETSc.DMDA().create([N_q1, N_q2],
                                dof = 3,
                                stencil_width = 0
                               )

vel_drift_vec   = vector_da.createGlobalVec()
vel_drift_array = vel_drift_vec.getArray()

PETSc.Object.setName(vel_drift_vec, 'vel_drift')




io = PetscBinaryIO.PetscBinaryIO()

dump_file = 'coords.bin'
coords = io.readBinaryFile(dump_file)
coords = coords[0].reshape(N_q2, N_q1, 13)

x = coords[:, :, 0]; y = coords[:, :, 1]

coords_vec   = scalar_da.getCoordinates()
coords_array = coords_vec.getArray()

coords_array[::2]  = x.flatten()
coords_array[1::2] = y.flatten()

scalar_da.setCoordinates(coords_vec)
vector_da.setCoordinates(coords_vec)

filepath = '.'

filename = "t=00000020.000000.bin"
dump_file = filepath+'/dump_moments/'+filename
moments = io.readBinaryFile(dump_file)
moments = moments[0].reshape(N_q2, N_q1, 3)

dump_file = filepath+'/dump_lagrange_multipliers/'+filename
lagrange_multipliers = io.readBinaryFile(dump_file)
lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 5)

density = moments[:, :, 0]
vel_drift_x  = lagrange_multipliers[:, :, 3]
vel_drift_y  = lagrange_multipliers[:, :, 4]

size = N_q1*N_q2

density_array[:] = density.flatten()

vel_drift_x_array[:] = vel_drift_x.flatten()
vel_drift_y_array[:] = vel_drift_y.flatten()

vel_drift_array[::3]  = vel_drift_x.flatten()
vel_drift_array[1::3] = vel_drift_y.flatten()
# third component is zero; do nothing

viewer = PETSc.Viewer().createVTK('ballistic_rectifier.vts', 'w')
density_vec.view(viewer)
vel_drift_x_vec.view(viewer)
vel_drift_y_vec.view(viewer)
vel_drift_vec.view(viewer)

