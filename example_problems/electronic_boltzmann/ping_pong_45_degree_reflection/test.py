import arrayfire as af
import numpy as np


# 2D array with dimensions 3x4 being theta and length respectively. 
A = af.Array([1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16], (3, 4))

A_1d = af.moddims(A, A.dims()[0]*A.dims()[1])

print ("A")
print (A)
print (A_1d)

# Aim : at each point in length, shift the array by a different value say [1, 2, 0, -1]
shifts = [1, 2, 0, -1]
shifts = af.Array(shifts) #Convert to af array

print ("Shifts")
print (shifts)


# Generate the shift_indices 2D array
temp = af.range(A.dims()[0])

shift_indices = 0.*A
index = 0
for value in shifts:
    print (value.scalar())
    shift_indices[:, index] = af.shift(temp.dims()[0]*index+temp, int(value.scalar()))
    index = index + 1


shift_indices_1d = af.moddims(shift_indices, shift_indices.dims()[0]*shift_indices.dims()[1])

print ("Shift indices")
print (shift_indices)
print (shift_indices_1d)


A_shifted_1d = A_1d[shift_indices_1d]
A_shifted = af.moddims(A_shifted_1d, A.dims()[0], A.dims()[1])

print ("Shifted array")
print (A_shifted_1d)
print (A_shifted)
