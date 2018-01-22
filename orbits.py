#####################################################################################
### Code by Lauren Hayward Sierens
#####################################################################################

from   itertools import product
import numpy as np
import random
from   scipy.optimize import fsolve

random.seed(111)

#       1 2 3 4 5 6 7 8 9 101112
dist= [[0,2,1,2,1,2,1,2,1,2,1,3], #1
       [2,0,2,2,1,1,2,2,1,1,3,1], #2
       [1,2,0,2,2,1,2,1,1,3,1,2], #3
       [2,2,2,0,2,2,1,1,3,1,1,1], #4
       [1,1,2,2,0,2,1,3,1,1,2,2], #5
       [2,1,1,2,2,0,3,1,1,2,2,1], #6
       [1,2,2,1,1,3,0,2,2,1,1,2], #7
       [2,2,1,1,3,1,2,0,2,2,1,1], #8
       [1,1,1,3,1,1,2,2,0,2,2,2], #9
       [2,1,3,1,1,2,1,2,2,0,2,1], #10
       [1,3,1,1,2,2,1,1,2,2,0,2], #11
       [3,1,2,1,2,1,2,1,2,1,2,0]] #12
dist = np.array(dist)

def incr_key(d,k):
  if d.get(k):
    d[k] += 1
  else:
    d[k] = 1

orbits={}

#Orbits with 0 or 12 points:
orbits[(0,())]  = 1
orbits[(12,())] = 1

#Orbits with 1 or 11 points:
orbits[(1,(0))] = 12
orbits[(11,(0))] = 12

#Orbits with 2 or 10 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    key = (dist[i1,i2])
    incr_key(orbits, (2,key))
    incr_key(orbits,(10,key))

#Orbits with 3 or 9 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    for i3 in range(i2+1, 12):
      key = tuple( sorted([ dist[i1,i2], dist[i1,i3], dist[i2,i3] ]) )
      incr_key(orbits, (3,key))
      incr_key(orbits,(9,key))

#Orbits with 4 or 8 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    for i3 in range(i2+1, 12):
      for i4 in range(i3+1, 12):
        key = tuple( sorted([ dist[i1,i2], dist[i1,i3], dist[i1,i4], dist[i2,i3], dist[i2,i4], dist[i3,i4] ]) )
        incr_key(orbits,(4,key))
        incr_key(orbits,(8,key))

#Orbits with 5 or 7 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    for i3 in range(i2+1, 12):
      for i4 in range(i3+1, 12):
        for i5 in range(i4+1, 12):
          key = tuple( sorted([ dist[i1,i2], dist[i1,i3], dist[i1,i4], dist[i1,i5],
                                             dist[i2,i3], dist[i2,i4], dist[i2,i5],
                                                          dist[i3,i4], dist[i3,i5],
                                                                       dist[i4,i5]]) )
          incr_key(orbits,(5,key))
          incr_key(orbits,(7,key))

#Orbits with 6 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    for i3 in range(i2+1, 12):
      for i4 in range(i3+1, 12):
        for i5 in range(i4+1, 12):
          for i6 in range(i5+1, 12):
            key = tuple( sorted([ dist[i1,i2], dist[i1,i3], dist[i1,i4], dist[i1,i5], dist[i1,i6],
                                               dist[i2,i3], dist[i2,i4], dist[i2,i5], dist[i2,i6],
                                                            dist[i3,i4], dist[i3,i5], dist[i3,i6],
                                                                         dist[i4,i5], dist[i4,i6],
                                                                                      dist[i5,i6]]) )
            incr_key(orbits,(6,key))

#Now let's label the orbits. Also, create the function f:
N_orbits = len(orbits)
dict_keyToLabel = {} #new dictionary with the same keys, but arguments are the labels
i=0
for key in orbits.keys():
  dict_keyToLabel[key] = i
  i = i+1

lst = list(product([0, 1], repeat=12))
configs = np.array(lst)
labels  = np.zeros(len(configs))

#Function to get the label corresponding to a given configuration:
def config_to_label(c):
  pts = [i for i in range(12) if c[i]==1]
  n = len(pts) #number of points
  
  if n>6:
    pts = [i for i in range(12) if c[i]==0]
  
  if n==0 or n==12:
    key = ()
  elif n==1 or n==11:
    key = (0)
  elif n==2 or n==10:
    key = (dist[pts[0],pts[1]])
  elif n==3 or n==9:
    key = tuple( sorted([ dist[pts[0],pts[1]], dist[pts[0],pts[2]], dist[pts[1],pts[2]] ]) )
  elif n==4 or n==8:
    key = tuple( sorted([ dist[pts[0],pts[1]], dist[pts[0],pts[2]], dist[pts[0],pts[3]],
                          dist[pts[1],pts[2]], dist[pts[1],pts[3]],
                          dist[pts[2],pts[3]] ]) )
  elif n==5 or n==7:
    key = tuple( sorted([ dist[pts[0],pts[1]], dist[pts[0],pts[2]], dist[pts[0],pts[3]], dist[pts[0],pts[4]],
                          dist[pts[1],pts[2]], dist[pts[1],pts[3]], dist[pts[1],pts[4]],
                          dist[pts[2],pts[3]], dist[pts[2],pts[4]],
                          dist[pts[3],pts[4]]]) )
  else: #n=6
    key = tuple( sorted([ dist[pts[0],pts[1]], dist[pts[0],pts[2]], dist[pts[0],pts[3]], dist[pts[0],pts[4]], dist[pts[0],pts[5]],
                          dist[pts[1],pts[2]], dist[pts[1],pts[3]], dist[pts[1],pts[4]], dist[pts[1],pts[5]],
                          dist[pts[2],pts[3]], dist[pts[2],pts[4]], dist[pts[2],pts[5]],
                          dist[pts[3],pts[4]], dist[pts[3],pts[5]],
                          dist[pts[4],pts[5]] ]) )

  return dict_keyToLabel[(n,key)]
#end config_to_label func

#This function converts an input config. (array) to a string (for file output):
def conf_to_str(c):
  s = ''
  for i in c:
    s = s + str(i)
  return s

#Loop over all configs to get all labels and write to file:
fout = open('config_labels.txt', 'w')
for (ic,c) in enumerate(configs):
  labels[ic] = config_to_label(c)
  fout.write('%s \t %d \n' %(conf_to_str(c),labels[ic]))
#end for
fout.close()

N = 2**12
px = 1.0/N

def psi(u,gam):
  return 1.0/ (1.0 + np.exp(-gam*u))

#Function that computes xlogx and gives 0 when x=0:
def xlogx(x):
  out = np.zeros(len(x))
  
  for i,xx in enumerate(x):
    if xx>1e-100:
     out[i] = xx*np.log2(xx)
  return out

#Function that fsolve will call:
def func((theta,gamma)):
  p_y1condx = psi( labels - theta, gamma )
  p_y0condx = 1.0 - p_y1condx
  p_y1 = px*np.sum(p_y1condx)
  
  cutoff = 1e-10
  p_y1 = min( p_y1, 1 - cutoff)
  p_y1 = max( p_y1, cutoff )
  
  p_y0 = 1.0 - p_y1
  
  #print '%f  %f' %(theta, gamma)
  
  MI = px*( np.sum( xlogx(p_y1condx) ) - np.sum( p_y1condx*np.log2(p_y1) )
          + np.sum( xlogx(p_y0condx) ) - np.sum( p_y0condx*np.log2(p_y0) ) )
           
  return [p_y1-0.5, (MI-0.99)*10]

th,ga = fsolve(func,[34,10])
out = func((th,ga))
print "\nFrom Fsolve: (theta, gamma) = (%f, %f)" %(th, ga)
print "p(y=1) = %f" %(out[0]+0.5)
print "MI     = %f" %((out[1]/10)+0.99)

print "\nAnna's solution: (theta, gamma) = (%f, %f)" %(34,30.5)
out = func((34,30.5))
print "p(y=1) = %f" %(out[0]+0.5)
print "MI     = %f" %((out[1]/10)+0.99)
