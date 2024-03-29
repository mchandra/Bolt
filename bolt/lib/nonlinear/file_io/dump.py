#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from petsc4py import PETSc
import numpy as np
import arrayfire as af
from bolt.lib.utils.af_petsc_conversion import af_to_petsc_glob_array

def dump_coordinate_info(self, arrays, name, file_name):

    self._da_coord_arrays = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                   dof        = len(arrays),
                                                   proc_sizes = (self._nproc_in_q1,
                                                                 self._nproc_in_q2
                                                                ),
                                                   ownership_ranges = self._ownership_ranges,
                                                   comm       = self._comm
                                                 )
    self._glob_coord       = self._da_coord_arrays.createGlobalVec()
    self._glob_coord_array = self._glob_coord.getArray()


    N_g = self.N_ghost

    for i in range(len(arrays)):
        if (i==0):
            array_to_dump = arrays[0][:, :,  N_g:-N_g, N_g:-N_g]
        else:
            array_to_dump = af.join(0, array_to_dump,
                                    arrays[i][:, :,
                                    N_g:-N_g,
                                    N_g:-N_g]
                                   )
    af.flat(array_to_dump).to_ndarray(self._glob_coord_array)
    PETSc.Object.setName(self._glob_coord, name)
    viewer = PETSc.Viewer().createBinary(file_name + '.bin', 'w', comm=self._comm)
    viewer(self._glob_coord)


def dump_aux_arrays(self, arrays, name, file_name):

    if (self.dump_aux_arrays_initial_call):
        self._da_aux_arrays = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                   dof        = len(arrays),
                                                   proc_sizes = (self._nproc_in_q1,
                                                                 self._nproc_in_q2
                                                                ),
                                                   ownership_ranges = self._ownership_ranges,
                                                   comm       = self._comm
                                                 )
        self._glob_aux       = self._da_aux_arrays.createGlobalVec()
        self._glob_aux_array = self._glob_aux.getArray()

        self.dump_aux_arrays_initial_call = 0

    N_g = self.N_ghost

    for i in range(len(arrays)):
        if (i==0):
            array_to_dump = arrays[0][:, :,  N_g:-N_g, N_g:-N_g]
        else:
            array_to_dump = af.join(0, array_to_dump,
                                    arrays[i][:, :,
                                    N_g:-N_g,
                                    N_g:-N_g]
                                   )
    af.flat(array_to_dump).to_ndarray(self._glob_aux_array)
    PETSc.Object.setName(self._glob_aux, name)
    viewer = PETSc.Viewer().createBinary(file_name + '.bin', 'w', comm=self._comm)
    viewer(self._glob_aux)

def dump_moments(self, file_name):
    """
    This function is used to dump moment variables to a file for later usage.

    Parameters
    ----------

    file_name : str
                The variables will be dumped to this provided file name.

    Returns
    -------

    This function returns None. However it creates a file 'file_name.h5',
    containing all the moments that were defined under moments_defs in
    physical_system.

    Examples
    --------

    >> solver.dump_variables('boltzmann_moments_dump')

    The above set of statements will create a HDF5 file which contains the
    all the moments which have been defined in the physical_system object.
    The data is always stored with the key 'moments' inside the HDF5 file.
    Suppose 'density', 'mom_v1_bulk' and 'energy' are the 3 functions defined.
    Then the moments get stored in alphabetical order, ie. 'density', 'energy'...:

    These variables can then be accessed from the file using
    
    >> import h5py
    
    >> h5f    = h5py.File('boltzmann_moments_dump.h5', 'r')
    
    >> n      = h5f['moments'][:][:, :, 0]
    
    >> energy = h5f['moments'][:][:, :, 1]
    
    >> mom_v1 = h5f['moments'][:][:, :, 2]
    
    >> h5f.close()

    However, in the case of multiple species, the following holds:

    >> n_species_1      = h5f['moments'][:][:, :, 0]
 
    >> n_species_2      = h5f['moments'][:][:, :, 1]
    
    >> energy_species_1 = h5f['moments'][:][:, :, 2]

    >> energy_species_2 = h5f['moments'][:][:, :, 3]
    
    >> mom_v1_species_1 = h5f['moments'][:][:, :, 4]

    >> mom_v1_species_2 = h5f['moments'][:][:, :, 5]
    """
    N_g = self.N_ghost

    attributes = [a for a in dir(self.physical_system.moments) if not a.startswith('_')]

    # Removing utility functions and imported modules:
    if('integral_over_p' in attributes):
        attributes.remove('integral_over_p')
    if('params' in attributes):
        attributes.remove('params')

    for i in range(len(attributes)):
        #print("i = ", i, attributes[i])
        if(i == 0):
            array_to_dump = self.compute_moments(attributes[i])
        else:
            array_to_dump = af.join(1, array_to_dump,
                                    self.compute_moments(attributes[i])
                                   )

        af.eval(array_to_dump)

    
    af_to_petsc_glob_array(self, array_to_dump, self._glob_moments_array)

    viewer = PETSc.Viewer().createBinary(file_name + '.bin', 'w', comm=self._comm)
    viewer(self._glob_moments)

def dump_distribution_function(self, file_name):
    """
    This function is used to dump distribution function to a file for
    later usage.This dumps the complete 5D distribution function which
    can be used for restarting / post-processing

    Parameters
    ----------

    file_name : The distribution_function array will be dumped to this
                provided file name.

    Returns
    -------

    This function returns None. However it creates a file 'file_name.h5',
    containing the data of the distribution function

    Examples
    --------
    
    >> solver.dump_distribution_function('distribution_function')

    The above statement will create a HDF5 file which contains the
    distribution function. The data is always stored with the key 
    'distribution_function'

    This can later be accessed using

    >> import h5py
    
    >> h5f = h5py.File('distribution_function.h5', 'r')
    
    >> f   = h5f['distribution_function'][:]
    
    >> h5f.close()

    Alternatively, it can also be used with the load function to resume
    a long-running calculation.

    >> solver.load_distribution_function('distribution_function')
    """

    af_to_petsc_glob_array(self, self.f, self._glob_f_array)
    viewer = PETSc.Viewer().createBinary(file_name + '.bin', 'w', comm=self._comm)
    viewer(self._glob_f)

    return

def dump_EM_fields(self, file_name):
    """
    This function is used to EM fields to a file for later usage.
    This dumps all the EM fields quantities E1, E2, E3, B1, B2, B3 
    which can then be used later for post-processing

    Parameters
    ----------

    file_name : The EM_fields array will be dumped to this
                provided file name.

    Returns
    -------

    This function returns None. However it creates a file 'file_name.h5',
    containing the data of the EM fields.

    Examples
    --------
    
    >> solver.dump_EM_fields('data_EM_fields')

    The above statement will create a HDF5 file which contains the
    EM fields data. The data is always stored with the key 
    'EM_fields'

    This can later be accessed using

    >> import h5py
    
    >> h5f = h5py.File('data_EM_fields.h5', 'r')
    
    >> EM_fields = h5f['EM_fields'][:]

    >> E1 = EM_fields[:, :, 0]
    
    >> E2 = EM_fields[:, :, 1]
    
    >> E3 = EM_fields[:, :, 2]
    
    >> B1 = EM_fields[:, :, 3]
    
    >> B2 = EM_fields[:, :, 4]
    
    >> B3 = EM_fields[:, :, 5]

    >> h5f.close()

    Alternatively, it can also be used with the load function to resume
    a long-running calculation.

    >> solver.load_EM_fields('data_EM_fields')
    """
    
    af_to_petsc_glob_array(self, 
                           self.fields_solver.yee_grid_EM_fields,
                           self.fields_solver._glob_fields_array
                          )

    viewer = PETSc.Viewer().createBinary(file_name + '.bin', 'w', comm=self._comm)
    viewer(self.fields_solver._glob_fields)

    return
