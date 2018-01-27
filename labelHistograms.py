#####################################################################################
#### Code by Lauren Hayward Sierens
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np

#tag = 'random2'
#filename = 'config_labels_%s.txt' %(tag)

tag='carlos'
filename='LabelsCarlos.npy'

theta=0
gamma=0
if filename == 'config_labels_python2.txt':
  theta=34.0
  gamma=30.5
elif filename == 'config_labels_python3.txt':
  theta=39.1
  gamma=30.0
elif filename == 'config_labels_random1.txt':
  theta=35.05
  gamma=50.0
elif filename == 'config_labels_random2.txt':
  theta=32.2
  gamma=15

def step(u):
  out = np.array([1.0,0]) #0
  
  if u>0:
    out = np.array([0,1.0]) #1
  return out

def psi(u,gam):
  return 1.0/ (1.0 + np.exp(np.minimum(-gam*u,700)))

N=12
N_data=2**N

hist_y0_data = []
hist_y1_data = []
num_y1 = 0

if (theta>0 and gamma>0):
  fin = open(filename, 'r')
  x_data = np.zeros((N_data,N))
  y_data = [None for i in range(N_data)]

  count = 0
  for line in fin:
    tokens = line.split()
    c_str = tokens[0]
    
    for i in range(len(c_str)):
      x_data[count,i] = int(c_str[i])
    
    fx = int(tokens[1])
    y_data[count] = step( fx - theta )
    if y_data[count][0] == 0:
      hist_y0_data.append(np.sum(x_data[count])-0.5)
    else:
      hist_y1_data.append(np.sum(x_data[count])-0.5)
      num_y1 = num_y1 + 1

    count = count + 1
  fin.close()
elif filename == 'LabelsCarlos.npy':
  configs = np.array(np.load('configs.npy'))
  labels = np.array(np.load(filename))
  
  for c,l in zip(configs,labels):
    if l[0] == 0:
      hist_y0_data.append(np.sum(c)-0.5)
    else:
      hist_y1_data.append(np.sum(c)-0.5)
      num_y1 = num_y1 + 1

hist_all     = hist_y0_data + hist_y1_data
hist_y0_data = np.array(hist_y0_data)
hist_y1_data = np.array(hist_y1_data)
hist_all     = np.array(hist_all)

print("Number with y=1: %d out of %d" %(num_y1,N_data))
bins_arr = np.array(range(N+2))-0.5

plt.hist(hist_all,    bins=bins_arr,color=(0.75,0.75,0.75), label='all configs.')
plt.hist(hist_y0_data,bins=bins_arr, edgecolor='black',     label='y=0')
plt.legend()
plt.savefig('histo_y0_%s.pdf' %tag)

plt.figure()
plt.hist(hist_all,    bins=bins_arr,color=(0.75,0.75,0.75), label='all configs.')
plt.hist(hist_y1_data,bins=bins_arr, edgecolor='black',     label='y=1')
plt.legend()
plt.savefig('histo_y1_%s.pdf' %tag)

#plt.show()

