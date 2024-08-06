"""
Plot the ball trajectory in 2D e.g Y vs Z
"""

import numpy as np
import matplotlib.pyplot as plt

# load ball trajectory, '[x y z]' on each line of file (including square brackets)
data = np.array([np.fromstring(line.replace("[", "").replace("]", ""), sep=' ') for line in open('balltraj.txt')])


# plot y vs z
plt.plot(data[:,1], data[:,2])
plt.xlabel('Y')
plt.ylabel('Z')

# save
plt.savefig('ball_traj.png')


