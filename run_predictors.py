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
	print('\t\t\tAccuracy\t\tTime')
	
	# create storage
	names, accs, times = [], [], []
	
	
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	
	# weighted effective transition Alt
	start = time.time()
	pred = weightedEffectiveTransitionAlt(G)
	acc = pred.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Weighted ET Alt\t\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Weighted\nET')
	accs.append(acc)
	times.append(end)	
	
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	
	# weighted effective transition Alt
	start = time.time()
	pred7 = weightedEffectiveTransitionApprox(G)
	acc = pred7.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Weighted ET Approx\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Weighted\nET')
	accs.append(acc)
	times.append(end)	
	
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	
	# shortest path
	start = time.time()
	pred1 = op.shortestPath(G)
	acc = pred1.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Shortest Path\t\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Shortest\nPath')
	accs.append(acc)
	times.append(end)	
	
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	
	# common neighbors
	start = time.time()
	pred2 = op.commonNeighbors(G)
	acc = pred2.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Common Neighbors\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Common\nNeighbors')
	accs.append(acc)
	times.append(end)	
	
	"""
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	# effective transition
	start = time.time()
	pred = effectiveTransition(G)
	acc = pred.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Effective Transition\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Effective\nTransition')
	accs.append(acc)
	times.append(end)	
	"""
	
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()

	# effective transition Alt
	start = time.time()
	pred3 = effectiveTransitionAlt(G)
	acc = pred3.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Effective Tran Alt\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Effective\nTransition')
	accs.append(acc)
	times.append(end)	
	
	"""
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()

	
	# effective transition Approx
	start = time.time()
	pred8 = effectiveTransitionApprox(G)
	acc = pred8.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Effective Tran Approx\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Effective\nTransition')
	accs.append(acc)
	times.append(end)	
	"""
	
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	
	# spectral embedding
	start = time.time()
	pred4 = se.spectralEmbed(G, dim=4)
	acc = pred4.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Spectral Embed\t\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Spectral\nEmbed')
	accs.append(acc)
	times.append(end)	
	
	"""
	G = H.copy()
	to_predict = to_predict_.copy()
	fullG = fullG_.copy()
	
	# weighted effective transition
	start = time.time()
	pred = weightedEffectiveTransition(G)
	acc = pred.accuracy(true_graph=fullG, k=len(to_predict))
	end = time.time() - start
	print('Weighted ET\t\t{}\t\t\t{}'.format(round(acc,5),round(end,5)))
	names.append('Weighted\nET')
	accs.append(acc)
	times.append(end)	
	"""

	


	
	return names, accs, times	

#filenames = ['haggledata','hypertextdata','infectiousdata','hepph','hepth','facebookWSON','facebook','internet','reality_miningdata']
#filenames = ['haggledata','hypertextdata','infectiousdata','reality_miningdata']
#filenames = ['hypertextdata','reality_miningdata']
filenames = ['haggledata','hypertextdata','reality_miningdata']

for fname in filenames:
	nms, accs, tms = get_predictions(fname)
	
	d = {'':nms,'acc':accs,'timer':tms}
	df = pd.DataFrame(data=d)
	df = df.set_index('')
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax2 = ax.twinx()
	
	width = .4
	a = df.acc.plot.bar(color='black',ax=ax, width=width,position=1,label='Accuracy',rot=0)
	b = df.timer.plot(kind='bar',color='slategrey', ax=ax2,width=width,position=0,label='Time')
	
	ax.set_ylabel('Accuracy')
	ax2.set_ylabel('Time')
	ax.legend(loc='upper left')
	ax2.legend(loc='upper right')
	plt.title(fname)
	plt.xlabel('')
	plt.savefig('results/'+fname+'_results.png')
	
	
	
