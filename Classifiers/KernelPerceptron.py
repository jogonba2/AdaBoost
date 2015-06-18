#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Overxfl0w13 #
 
"""You can define more kernel functions to test its performance whenever it respect the following condition: 
  http://latex.codecogs.com/gif.latex?\forall i\in K(x,y)\rightarrow i\geq 0
  Some examples: http://gyazo.com/3b1d3ae355c2638f5ac4d98c82c31d12 (Theme 4: representation based on kernels.) Perception (PER), DSIC-UPV)
"""

# Kernel test -> hamming distance #
def kernel_sample_d_hamming(x,y):
	if len(x)!=len(y): 
		print "Aborting, not same dimension ",x,y
		exit()
	else: 
		s=0
		for d in xrange(len(x)): s+=abs(x[d]-y[d])
		return 1/(s+1)
		
def perceptron_train(train_samples,kernel):
	alpha,counter_end = [0 for x in xrange(len(train_samples))],0
	while counter_end!=len(train_samples):
		counterx,counter_end = 0,0
		for x in train_samples:
			gx,counterxi = 0,0
			for xi in train_samples: gx += alpha[counterxi]*xi[1]*kernel(x[0],xi[0]) + alpha[counterxi]*xi[1]; counterxi+=1
			if x[1]*gx<=0: alpha[counterx]+=1;
			else: counter_end += 1
			counterx += 1
	return alpha	

def perceptron_recog(train_samples,alpha,sample,kernel):
	gx,counterxi = 0,0
	for xi in train_samples: gx += alpha[counterxi]*xi[1]*kernel(sample,xi[0]) + alpha[counterxi]*xi[1]; counterxi+=1
	return 1 if gx>=0 else -1

def classify(train_samples,sample):
	alpha  = perceptron_train(train_samples,kernel_sample_d_hamming)
	cclass = perceptron_recog(train_samples,alpha,sample,kernel_sample_d_hamming)
	return cclass
	
"""if __name__ == '__main__':
	train_samples = [([1,1],1),([2,2],-1),([1,3],1),([3,1],-1),([3,3],1)]
	sample_test = [3,1]
	print classify(train_samples,sample_test)
"""
