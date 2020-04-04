import numpy as np

q1_0 = 0.
q2_0 = 0.

v_f = 1.
t   = 10. #ps

theta = 0.7609

q1 = v_f*np.cos(theta)*t
q2 = v_f*np.sin(theta)*t

x_pos = q1
y_pos = q2

while (x_pos > 2):
    x_pos = x_pos - 2
while (y_pos > 2):
    y_pos = y_pos - 2

print (q1, q2, np.sqrt(q1**2 + q2**2))
print (x_pos, y_pos)

