#!/usr/bin/env python
# -*- coding: utf-8 -*-

def vectorxvector(v,y):
	s = 0
	for n in xrange(len(v)):
		s += v[n] * y[n]
	return s
	
def vectorxconstant(v,a):
	return [x*a for x in v]

def vectordifference(v,y):
	return [v[n]-y[n] for n in xrange(len(v))]

def vectorsum(v,y):
	return [v[n]+y[n] for n in xrange(len(v))]	

def classify(learnSamples,recognizementSample,b=0.00001,alpha=0.00001,maxLimit=1000):
	m             = 0;
	c             = 0;
	weightVectors = []
	for i in xrange(2): weightVectors.append([0 for x in xrange(len(learnSamples[0]))])
	while(m<len(learnSamples) and c<maxLimit):
		m = 0 
		c += 1
		for n in xrange(len(learnSamples)):
			i = learnSamples[n][1]
			if i==-1: i=0
			g = vectorxvector(weightVectors[i],learnSamples[n][0])
			error = False
			for j in xrange(len(weightVectors)):
				if(j!=i):
					if(vectorxvector(weightVectors[j],learnSamples[n][0]) + b > g):
						weightVectors[j] = vectordifference(weightVectors[j],vectorxconstant(learnSamples[n][0],alpha))
						error = True
			if error: weightVectors[i] = vectorsum(weightVectors[i],vectorxconstant(learnSamples[n][0],alpha))
			else: m += 1	
	recognizeds = []
	maxim,i = -float('inf'),0
	for iv in xrange(len(weightVectors)):
		prod = vectorxvector(weightVectors[iv],recognizementSample)
		if prod > maxim:
			maxim = prod
			i = iv
	if i==0: i=-1
	return i
	
"""if __name__ == '__main__':
	learnSamples = [([1,1],-1),([2,2],1),([0.5,0.7],1)]
	recognizementSample = [-5,-7]
	print classify(learnSamples,recognizementSample)
	#recognizeds = perceptron_recognizement()		
	#plotSamplesRecognizeds(recognizeds)
"""
