To switch between 1D/2D polar and cartesian momentum space, the following changes need
to be made : 


1. domain
    - Change p2_start, p2_end and N_p2
2. params
    - Change p_space_grid
    - Change p_dim (if switching between 1D and 2D representation in p-space)
3. initialize : No change
4. main : No change
i5. boundary_conditions : No change



PdCoO2 works only with the polar2D grid in momentum space for the moment.

To change the extention regions of the wire, just change q1/q2 start/end in domain and manually adjust domain allocation in params. Divide the entire extension domain on one side of the circular region into 3 vertical patches (and not 6 for example).
