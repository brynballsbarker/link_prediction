import other_predictors as op
import numpy as np
import time
import networkx as nx
from load_data import load_data
from effective_transition import effectiveTransition, weightedEffectiveTransition
from effective_transition import effectiveTransitionAlt, weightedEffectiveTransitionAlt
from effective_transition_approx import effectiveTransitionApprox, weightedEffectiveTransitionApprox
import matplotlib.pyplot as plt
import spectral_embed as se
import pandas as pd

def get_predictions(fname):
	filename = 'data/'+fname+'.txt'
	H, to_predict_, fullG_ = load_data(filename)
	print('\n%s' % fname)
	print(len(H.nodes()))
	print('Steps\t\t\tAccuracy\t\tTime')
	
	
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	
	# weighted effective transition Alt
	start = time.time()
	pred = weightedEffectiveTransitionAlt(G)
	acc = pred.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Actual\t\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))

	
	for i in range(1,51):
	
		G = H.copy()
		to_predict = to_predict_.copy()
		fullG = fullG_.copy()
	
		# weighted effective transition Alt
		start = time.time()
		pred7 = weightedEffectiveTransitionApprox(G, cut=i)
		acc = pred7.accuracy(true_graph=fullG, k=len(to_predict))
		end = time.time() - start
		print('{}\t\t{}\t\t\t{}'.format(i,round(acc,5),round(end,5)))

#filenames = ['haggledata','hypertextdata','infectiousdata','hepph','hepth','facebookWSON','facebook','internet','reality_miningdata']
#filenames = ['haggledata','hypertextdata','infectiousdata','reality_miningdata']
filenames = ['hypertextdata','reality_miningdata']
#filenames = ['haggledata','hypertextdata','reality_miningdata']

for fname in filenames:
	get_predictions(fname)

	
	
	
