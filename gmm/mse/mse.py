import numpy as np
import sklearn as skl

from sklearn import mixture


cbow = np.load('gmm_2089811_cbow.npy').item()['covariances']
sg = np.load('gmm_2089811_skipgram.npy').item()['covariances']

print('cbow')
# MSE for the cbow model
for i in range(50):
	print(np.sum(np.diag(cbow[i])))

print('skipgram')
# MSE for skipgram model
for i in range(50):
	print(np.sum(np.diag(sg[i])))
