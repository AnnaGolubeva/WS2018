#####################################################################################
#### Code by Lauren Hayward Sierens
#####################################################################################

from   itertools import product
import numpy as np
import random
from   scipy.optimize import fsolve

seed=111
random.seed(seed)
np.random.seed(seed)

filename = 'config_labels_python2.txt'
fin = open(filename, 'r')

theta=0
gamma=0
if filename == 'config_labels_python2.txt':
  theta=34.0
  gamma=30.5
elif filename == 'config_labels_python3.txt':
  theta=39.1
  gamma=30.0

def step(u):
  out = np.array([1.0,0]) #0
  
  if u>0:
    out = np.array([0,1.0]) #1
  return out

def psi(u,gam):
  return 1.0/ (1.0 + np.exp(np.minimum(-gam*u,700)))

N_data=2**12
x_data = np.zeros((N_data,12))
y_data = [None for i in range(N_data)]
count = 0
num_y1 = 0
for line in fin:
  tokens = line.split()
  c_str = tokens[0]
  
  for i in range(len(c_str)):
    x_data[count,i] = int(c_str[i])
  
  fx = int(tokens[1])
  y_data[count] = step( fx - theta )
  if y_data[count][0] == 1:
      num_y1 = num_y1 + 1

  count = count + 1
fin.close()

print("Number with y=1: %d out of %d" %(num_y1,N_data))




