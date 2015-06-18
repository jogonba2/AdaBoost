#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Overxfl0w13 #
# Multinomial classifier #

from math import log

def laplace_smoothing(sequence_probability,epsilon):
	for l in sequence_probability:
		aux = 0
		for x in xrange(len(l)): l[x] += epsilon; aux += l[x]
		for x in xrange(len(l)): l[x] /= aux
	return sequence_probability
	
def class_probability(samples):
	res = {}
	for sample in samples: res[sample[1]] = 0 if sample[1] not in res else res[sample[1]]
	for sample in samples: res[sample[1]] += 1
	return [float(x)/len(samples) for x in res.values()]

def sequence_probability(samples,classes):
	res = [[] for c in classes]
	for c in xrange(len(classes)):
		c_samples = extract_class_samples(samples,classes[c])
		s,aux     = c_samples[0][0],0
		for x in xrange(1,len(c_samples)): s = vector_sum(s,c_samples[x][0]) 
		for d in s: aux += d
		res[c] = constant_x_vector((1.0/aux),s)
	return res

def classify(samples,sample,classes=[-1,1],epsilon=0.1):
	class_prob = class_probability(samples)
	sequence_prob = laplace_smoothing(sequence_probability(samples,classes),epsilon)
	c_max,v_max = -100,float("-inf")
	for y in xrange(len(class_prob)):
		cx   = vector_x_vector(log_vector(sequence_prob[y]),sample)+log(class_prob[y],2)
		if cx > v_max: 
			v_max=cx
			c_max = y
	return classes[c_max]
	
def extract_class_samples(samples,c): return [sample for sample in samples if sample[1]==c]
def vector_sum(v1,v2): return [v1[i]+v2[i] for i in xrange(len(v1))]
def constant_x_vector(a,v): return [a*v[i] for i in xrange(len(v))]
def log_vector(v): return[log(v[i],2) for i in xrange(len(v))]
def vector_x_vector(v1,v2): return reduce(lambda x,y: x+y,[v1[i]*v2[i] for i in xrange(len(v1))])

def __str__(): return "Multinomial classifier"
"""if __name__ == "__main__":
	samples = [([1,1],-1),([3,3],-1),([2,100],1),([2,20],1)]
	sample_test   = [1,1]
	classified    = classify(samples,sample_test)
	print "Correct class of sample: ",sample_test," is: ", classified
"""
