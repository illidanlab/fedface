import numpy as np
import scipy.io as sio
import h5py
# import hdf5storage
import os
import csv 
import re

# def saveMatv7(fname, data, version=None):
# 	path = os.path.dirname(fname)
# 	name = os.path.basename(fname)
# 	hdf5storage.write(data, path, fname, matlab_compatible=True)

def load_data(filename, delimiter=r'[ ,\t]+'):
	with open(filename, 'r') as f:
		lines = f.readlines()
	lines = [re.split(delimiter, line.strip()) for line in lines]

	return np.array(lines, dtype=np.object)

def load_mat(filename):
	try:
		return sio.loadmat(filename)
	except:
		dataset = {}
		with h5py.File(filename, 'r') as f:
			for k in f.keys():
				dataset[k] = np.array(f[k])
		return dataset
