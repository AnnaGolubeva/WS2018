#####################################################################################
### Code by Lauren Hayward Sierens
#####################################################################################

import numpy as np

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

#Orbits with 0 points:
orbits[()] = 1

#Orbits with 1 point:
orbits[(0)] = 12

#Orbits with 2 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    key = (dist[i1,i2])
    incr_key(orbits,key)

#Orbits with 3 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    for i3 in range(i2+1, 12):
      key = tuple( sorted([ dist[i1,i2], dist[i1,i3], dist[i2,i3] ]) )
      incr_key(orbits,key)

#Orbits with 4 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    for i3 in range(i2+1, 12):
      for i4 in range(i3+1, 12):
        key = tuple( sorted([ dist[i1,i2], dist[i1,i3], dist[i1,i4], dist[i2,i3], dist[i2,i4], dist[i3,i4] ]) )
        incr_key(orbits,key)

#Orbits with 5 points:
for i1 in range(12):
  for i2 in range(i1+1, 12):
    for i3 in range(i2+1, 12):
      for i4 in range(i3+1, 12):
        for i5 in range(i4+1, 12):
          key = tuple( sorted([ dist[i1,i2], dist[i1,i3], dist[i1,i4], dist[i1,i5],
                                             dist[i2,i3], dist[i2,i4], dist[i2,i5],
                                                          dist[i3,i4], dist[i3,i5],
                                                                       dist[i4,i5]]) )
          incr_key(orbits,key)

L1 = len(orbits)

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
          incr_key(orbits,key)

L2 = len(orbits) - L1
print orbits
print (2*L1 + L2)

