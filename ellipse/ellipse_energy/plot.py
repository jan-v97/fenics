#!/usr/bin/python
import numpy as np
import argparse
from math import *

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



#read input
fobj = open("C:\\Users\\janve\\local\\masterarbeit\\fenics\\test\\ellipse_energy\\output.txt")
maxrad = float(fobj.readline())
minrad = float(fobj.readline())
radsteps = int(fobj.readline())
rotsteps = int(fobj.readline())
circle = float(fobj.readline())

string = ''
loc =''

Energie = np.zeros((radsteps*(rotsteps+1)),dtype = float)
x = np.zeros((radsteps*(rotsteps+1)),dtype = float)
y = np.zeros((radsteps*(rotsteps+1)),dtype = float)

# print(radsteps*rotsteps)

k=0
while ((loc != '\n')&(k<=(rotsteps)*(radsteps))):
	loc = fobj.read(1)
	if (loc == '\n'):
		break
	i = k//(radsteps)
	j = k%(radsteps)
	a1 = maxrad - i*(maxrad-minrad)/(radsteps)
	a2 = j*(pi/(2*(rotsteps-1)))
	x[k] = a1
	y[k] = a2
	#x[k] = a1 * sin(a2)
	#y[k] = a1 * cos(a2)
	if (loc != ' '):
		string += loc
	else:
		#print (string)
		Energie[k] = float(string)
		print (i," ",j, " ",k, " " ,Energie[k], '\n')
		string = ''
		k += 1

for i in range(0,rotsteps):
	a2 = a2 = i*(pi/(2*(rotsteps-1)))
	x[radsteps*rotsteps+i] = 1
	y[radsteps*rotsteps+i] = a2
	#x[radsteps*rotsteps+i] = sin(a2)
	#y[radsteps*rotsteps+i] = cos(a2)
	Energie[radsteps*rotsteps+i] = circle

print (Energie)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(y, x, Energie)
fig.show()
fig.savefig("C:\\Users\\janve\\local\\masterarbeit\\fenics\\test\\ellipse_energy\\output_pic")
print ("output: ", "output_pic")
plt.clf()


