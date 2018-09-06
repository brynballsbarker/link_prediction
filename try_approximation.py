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

fname = input('what file\n')
filename = 'data/'+fname+'.txt'
G, to_predict, fullG = load_data(filename)
print('\n%s' % fname)
print(len(G.nodes()))
cut = int(input('when to cut the sum\n'))

# weighted effective transition Alt
start = time.time()
pred = weightedEffectiveTransitionApprox(G, cut=cut)
acc = pred.accuracy(true_graph=fullG, k=len(to_predict))
end = time.time() - start
print('{}\t\t{}\t\t\t{}'.format(cut,round(acc,5),round(end,5)))

#filenames = ['haggledata','hypertextdata','infectiousdata','hepph','hepth','facebookWSON','facebook','internet','reality_miningdata']
#filenames = ['haggledata','hypertextdata','infectiousdata','reality_miningdata']
#filenames = ['hypertextdata','reality_miningdata']

#filenames = ['haggledata','hypertextdata','reality_miningdata']


	
	
	
