#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  KNN.py
#  
#  Copyright 2015 Overxflow 
#  

""" Vector difference """
def v_difference(x,y): return [abs(x[i]-y[i]) for i in xrange(len(x))]

""" Vector difference for weighted euclidean distance """
def v_pond_difference(x,y,w): return [w*abs(x[i]-y[i]) for i in xrange(len(x))]

""" Returns index of max first element in list inside another list """
def max_distance(l): 
	i,m,mp = 0,float("-inf"),0
	for sl in l:
		if sl[0]>m:
			m  = sl[0]
			mp = i
		i += 1
	return (mp,m)

""" Defines p-distance family """
def p_distance_function(p,x,y,w):
	if p==0:     return max(map(abs,v_difference(x,y)))
	else:    
		if w<=0: return (sum(v_difference(x,y)))**(1.0/p) # If w<=0, distance function isn't a metric.
		else:    return (sum(v_difference(x,y,w)))**(1.0/p)

""" Wilson prototype edition algorithm """
def wilson(prototypes,p,k,w):
	error,new_prototypes = 1,[]
	while error:
		error = 0
		for i in xrange(len(prototypes)):
			prototype = prototypes[i]
			c         = prototype[1]
			cp = knn(prototypes,p,k,prototype[0],w)
			if cp==c: new_prototypes.append(prototype)			
			else: error=1; prototypes.remove(prototype); break; # Slowly method :( #
	return prototypes

""" Condensed nearest neighbours algorithm """	
def cnn(prototypes,p,k,w):
	S,G = [prototypes[0]],[]
	# First phase #
	for i in xrange(1,len(prototypes)):
		prototype = prototypes[i]
		c         = prototype[1]
		cp = knn(S,p,k,prototype[0],w)
		if cp!=c: S.append(prototype)
		else: G.append(prototype)
	# Second phase #
	error = 1
	while G!=[] and error==1:
		error = 0
		for prototype in G:
			cp = knn(S,p,k,prototype[0],w)
			if cp!=prototype[1]: S.append(prototype); G.remove(prototype); error = 1; break;
	return S
	
	
""" K nearest neighbours algorithm """
def knn(prototypes,p,k,test_sample,w):
	classes = []
	k_nearest,l= [],0
	y = test_sample
	for prototype in prototypes:
		c         = prototype[1]
		prototype = prototype[0]
		distance  = p_distance_function(p,prototype,y,w)
		if l<k: k_nearest.append([distance,c])
		else: 
			(pos_max,max_dist) = max_distance(k_nearest)
			if distance<max_dist: 
				k_nearest[pos_max] = [distance,c]
		l += 1
	h = {}
	for nearest in k_nearest:
		if nearest[1] not in h: h[nearest[1]] = 1
		else: h[nearest[1]] += 1
	return max(h,key=h.get)
	

""" Core k nearest neighbours """
def classify(prototypes,test_sample,p=2,k=1,w=0,wil=1,cn=1):
	# Make Wilson edition #
	if wil==1: prototypes = wilson(prototypes,p,k,w)
	# Make CNN #
	if  cn==1: prototypes = cnn(prototypes,p,k,w)
	# Classificate KNN #
	return knn(prototypes,p,k,test_sample,w)
	
def __str__(): return "K_nearest_neighbour classifier"
	
"""if __name__ == "__main__":
	prototypes    = [([1,2],-1),([1,3],-1),([2,2],-1),([2,3],-1),([5,2],1),([5,3],1),([5,4],1),([4,3],1),([4,4],1)] # Train samples #
	test_sample  = [1,3] # Test samples #
	print classify(prototypes,test_sample,2,1)
"""
