import other_predictors as op
import numpy as np
import time
import networkx as nx
from load_data import load_data
from effective_transition import effectiveTransition, weightedEffectiveTransition
import matplotlib.pyplot as plt
import spectral_embed as se
import pandas as pd

def get_predictions(fname):
	filename = 'data/'+fname+'.txt'
	G, to_predict, fullG = load_data(filename)
	print('\n%s' % fname)
	print(len(G.nodes()))

filenames = ['haggledata','hypertextdata','infectiousdata','hepph','hepth','facebookWSON','facebook','internet','reality_miningdata']
#filenames = ['haggledata','hypertextdata','infectiousdata','reality_miningdata']
#filenames = ['hypertextdata','reality_miningdata']

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
	
	
	
