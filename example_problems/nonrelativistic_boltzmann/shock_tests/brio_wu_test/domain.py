from params import L_x, L_y, v_max

q1_start = 0
q1_end   = L_x
N_q1     = 1024

q2_start = 0
q2_end   = L_y
N_q2     = 3

p1_start = -v_max
p1_end   =  v_max
N_p1     = 20

p2_start = -v_max
p2_end   =  v_max
N_p2     = 20

p3_start = -v_max
p3_end   =  v_max
N_p3     = 20
    
N_ghost = 3
