To switch between 1D/2D polar and cartesian momentum space, the following changes need
to be made : 


1. domain
    - Change p2_start, p2_end and N_p2
2. params
    - Change p_space_grid
    - Change p_dim (if switching between 1D and 2D representation in p-space)
3. initialize : No change
4. main : No change
5. boundary_conditions : No change


-------------------


PdCoO2 works only with the polar2D grid in momentum space for the moment.


-------------------

List of changes made to enable magnetic field

1. params
    - fields_enabled = True
    - Set the value of l_c in params #see /src/electronic_boltzmann/advection_terms.py for details
