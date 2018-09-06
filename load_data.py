import numpy as np
from math import floor
import networkx as nx

def load_data(filename):
    myfile = open(filename, 'r')
    lines = myfile.readlines()
    lines = [line.split() for line in lines]
    #time = 3 if len(lines[0])==4 else 2
    lines = [[int(line[0]),int(line[1]),int(line[-1])] for line in lines]
    lines.sort(key=lambda x: x[2])
    n = len(lines)
    chop = floor(3*n/4)
    train = lines[:chop]
    to_train = [(edge[0],edge[1]) for edge in train]
    test = lines[chop:]
    k = len(test)
    to_predict = [(edge[0],edge[1]) for edge in test]
    G = nx.Graph()
    fullG = nx.Graph()
    G.add_edges_from(to_train)
    fullG.add_edges_from(to_train)
    G = max(nx.connected_component_subgraphs(G), key=len)
    fullG = max(nx.connected_component_subgraphs(fullG), key=len)
    nodes = G.nodes()
    to_predict = [(edge[0],edge[1]) for edge in test if edge[0] in nodes and edge[1] in nodes] 
    fullG.add_edges_from(to_predict)
    return G, to_predict, fullG
