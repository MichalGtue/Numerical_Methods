from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time


A = np.array([[2,1,3],[1,1,1],[3,1,6]])

B= np.array([[1,1,1],[2,1,3],[3,1,6]])

P = np.array([[0,1,0],[1,0,0],[0,0,1]])

print(P@B)