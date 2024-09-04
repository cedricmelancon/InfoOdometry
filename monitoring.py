from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('pdf')

my_data = genfromtxt('monitoring.csv', delimiter=' ')

fig, ax = plt.subplots()

ax.plot(my_data[:,0], my_data[:,1])
plt.savefig('flownet.pdf')

fig, ax = plt.subplots()

ax.plot(my_data[:,0], my_data[:,2])
plt.savefig('encoder.pdf')

fig, ax = plt.subplots()

ax.plot(my_data[:,0], my_data[:,3])
plt.savefig('transition.pdf')

fig, ax = plt.subplots()

ax.plot(my_data[:,0], my_data[:,4])
plt.savefig('pose.pdf')

fig, ax = plt.subplots()

ax.plot(my_data[:,0], my_data[:,5])
plt.savefig('total.pdf')