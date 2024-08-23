import numpy as np
m = 4
k = 1
n = 5
A = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

b_phases = []
for row in A:
    phase = 0
    for i in range(m,m+k):
        if row[i] == 1 and row[i+n] == 1:
            phase += 1
    b_phases.append(phase)
print(b_phases)
    