#!/usr/bin/env python
from Classifiers import K_nearest_neighbour
from Classifiers import Multinomial
from Classifiers import Perceptron
from Classifiers import KernelPerceptron
from math import log,floor,e

# Evitar tener que clasificar 2 veces en el proceso #
# Implementar cargado de fichero de datos serializado #
# Implementar clasificación usando el clasificador combinado #

# Traducir #
def AdaBoost(samples,M):
	weight_samples    = [1.0/len(samples) for sample in samples]
	classifiers       = [K_nearest_neighbour,Multinomial,Perceptron,KernelPerceptron]
	classifiers_error = [0 for x in classifiers]
	final_classifier  = []
	for it in xrange(M):
		best_classifier = K_nearest_neighbour # Random #
		index_sample    = 0
		for sample in samples:
			cclass = sample[1]
			sample = sample[0]
			index_classifier = 0
			for classifier in classifiers:
				computed_class = classifier.classify(samples,sample)
				if computed_class != cclass: classifiers_error[index_classifier] += weight_samples[index_sample]
				index_classifier += 1
			index_sample += 1
		print classifiers_error
		# Calcular el mejor clasificador (menor error) #
		min_error = min(classifiers_error)
		best_classifier = classifiers[classifiers_error.index(min_error)]
		# Si el error > 0.5 parar #
		if min_error > 0.5: print "No es necesario continuar, mejor error con los clasificadores por separado.\n"; exit(0)
		# Recalcular peso del clasificador #
		alpha_best_classifier = (1.0/2)*log((1-min_error)/(min_error+(1.0/10**20)))
		# Configurar clasificador de la iteracion actual #
		final_classifier.append((alpha_best_classifier,best_classifier))
		# Recalcular pesos de las muestras #
		index_sample = 0
		for sample in samples:
			cclass = sample[1]
			sample = sample[0]
			weight_samples[index_sample] = weight_samples[index_sample]*(e**(-cclass*alpha_best_classifier*best_classifier.classify(samples,sample)))
		# Normalizar pesos de las muestras #
		index_sample  = 0
		total_weights = sum(weight_samples) 
		weight_samples = map(lambda x:float(x)/sum(weight_samples),weight_samples)
	return final_classifier

def load_data(filename): pass
def classify(final_classificator,sample): pass

if __name__ == "__main__":
	train_samples     = [([1,3.7,-1.0,-1337],-1),
([1,3.9,-1.2,-1337],-1),
([1,5.1,-1.6,-1337],-1),
([1,4.5,-1.5,-1337],-1),
([1,4.5,-1.6,-1337],-1),
([1,4.7,-1.5,-1337],-1),
([1,4.4,-1.3,-1337],-1),
([1,4.1,-1.3,-1337],-1),
([1,4.0,-1.3,-1337],-1),
([1,4.4,-1.2,-1337],-1),
([1,4.6,-1.4,-1337],-1),
([1,4.0,-1.2,-1337],-1),
([1,3.3,-1.0,-1337],-1),
([1,4.2,-1.3,-1337],-1),
([1,4.2,-1.2,-1337],-1),
([1,4.2,-1.3,-1337],-1),
([1,4.3,-1.3,-1337],-1),
([1,3.0,-1.1,-1337],-1),
([1,4.1,-1.3,-1337],-1),
([1,6.0,1.5,-1337],1),
([1,5.1,-1.9,-1337],1),
([1,5.9,1.1,-1337],1),
([1,5.6,-1.8,-1337],1),
([1,5.8,1.2,-1337],1),
([1,6.6,1.1,-1337],1),
([1,4.5,-1.7,-1337],1),
([1,6.3,-1.8,-1337],1),
([1,5.8,-1.8,-1337],1),
([1,6.1,1.5,-1337],1),
([1,5.1,1.0,-1337],1),
([1,5.3,-1.9,-1337],1)]
	print AdaBoost(train_samples,3)

