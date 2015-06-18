#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  AdaBoost.py
#  
#  Copyright 2015 Overxflow 
#  

from Classifiers import K_nearest_neighbour
from Classifiers import Multinomial
from Classifiers import Perceptron
from Classifiers import KernelPerceptron
from math import log,floor,e
try: import cPickle as pickle
except: import pickle
from sys import argv

# Traducir #
# Implementar gestion de arguments #
def AdaBoost(samples,M):
	weight_samples    = [1.0/len(samples) for sample in samples]
	classifiers       = [K_nearest_neighbour,Multinomial,Perceptron,KernelPerceptron]
	classifiers_error = [0 for x in classifiers]
	final_classifier  = []
	for it in xrange(M):
		best_classifier       = K_nearest_neighbour # Random #
		index_best_classifier = 0 # Random #
		index_sample          = 0
		calculated_classes = [[] for classifier in classifiers]
		for sample in samples:
			cclass = sample[1]
			sample = sample[0]
			index_classifier = 0
			for classifier in classifiers:
				computed_class = classifier.classify(samples,sample)
				calculated_classes[index_classifier].append(computed_class)
				if computed_class != cclass: classifiers_error[index_classifier] += weight_samples[index_sample]
				index_classifier += 1
			index_sample += 1
		# Calcular el mejor clasificador (menor error) #
		min_error = min(classifiers_error)
		index_best_classifier = classifiers_error.index(min_error)
		best_classifier = classifiers[index_best_classifier]
		# Recalcular peso del clasificador #
		alpha_best_classifier = (1.0/2)*log((1-min_error)/(min_error+(1.0/10**20)))
		# Configurar clasificador de la iteracion actual #
		final_classifier.append((alpha_best_classifier,best_classifier))
		# Si el error > 0.5 parar #
		if min_error>0.5 or min_error==0:  print "Mejor error con los clasificadores por separado.\n";  return final_classifier
		# Recalcular pesos de las muestras #
		index_sample = 0
		for sample in samples:
			cclass = sample[1]
			sample = sample[0]
			weight_samples[index_sample] = weight_samples[index_sample]*(e**(-cclass*alpha_best_classifier*calculated_classes[index_best_classifier][index_sample]))
		# Normalizar pesos de las muestras #
		index_sample  = 0
		total_weights = sum(weight_samples) 
		weight_samples = map(lambda x:float(x)/sum(weight_samples),weight_samples)
	return final_classifier

def load_data(filename): 
	with open(filename,'rb') as fd: obj = pickle.load(fd)
	fd.close()
	return obj
	
def save_object(object,dest):
	with open(dest,'wb') as fd:
		pickle.dump(object,fd,pickle.HIGHEST_PROTOCOL)
	fd.close()	
	
def classify(final_classifier,samples,sample):
	val = 0
	for item in final_classifier: val = item[0]*item[1].classify(samples,sample)
	return -1 if val<0 else 1

def stringify_classifier(final_classifier):  
	for item in final_classifier:  print item[0],item[1].__str__()

def usage():
	print "Usage: AdaBoost.py train_data_file iterations [classify] [test_data_file] \n\n\
\ttrain_data_file -> Name of file with train data\n\
\titerations      -> process iterations\n\
\tclassify        -> Optional, specifies if you want to classify test data\n\
\ttest_data_file  -> Optional, only if you want to classify, specifies name of file with test data\n"
	
if __name__ == "__main__":
	if len(argv)<3: usage();exit(0)
	train_samples = load_data("flowerdata.dat")
	#test_samples  = load_data("flowerdata.dat") # Test with same train data, ... VERY OPTIMISTIC!! #
	final_classifier = AdaBoost(train_samples,3)
	print classify(final_classifier,train_samples,[1,6.1,1.5,-1337])
	stringify_classifier(final_classifier)

