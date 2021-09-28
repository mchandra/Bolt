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

For switched domain, functions defined in coords need to be passed through params.
Comment out import coords in advection_terms.py and df_dt.py. And get the functions from params.
