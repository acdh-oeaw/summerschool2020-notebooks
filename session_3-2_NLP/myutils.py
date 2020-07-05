import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20.0, 15.0)

import re
import os
import sys
import pathlib
import multiprocessing
import urllib.request
import zipfile
import lxml.etree
import networkx as nx
from random import shuffle

import gensim 
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA


def draw_words(model, words, pca=False, alternate=True, arrows=True, x1=3, x2=3, y1=3, y2=3, title=''):
    '''
	# function draw_words
	# ... reduces dimensionality of vectors of given words either with PCA or with t-SNE and draws the words into a diagram
	# @param word2vec model     to visualize vectors from
	# @param list     words     list of word strings to visualize
	# @param bool     pca       use PCA (True) or t-SNE (False) to reduce dimensionality 
	# @param bool     alternate use different color and label align for every second word
	# @param bool     arrows    use arrows to connect related words (items that are next to each other in list)
	# @param float    x1        x axis range (from)
	# @param float    x2        x axis range (to)
	# @param float    y1        y axis range (from)
	# @param float    y2        y axis range (to)
	# @param string   title     for diagram
    '''

    # get vectors for given words from model
    vectors = [model.wv[word] for word in words]
    if pca:
        pca = PCA(n_components=2, whiten=True)
        vectors2d = pca.fit(vectors).transform(vectors)
    else:
        tsne = TSNE(n_components=2, random_state=0)
        vectors2d = tsne.fit_transform(vectors)

    # draw image
    plt.figure(figsize=(15,15))
    if pca:
        plt.axis([x1, x2, y1, y2])

    first = True # color alternation to divide given groups
    for point, word in zip(vectors2d , words):
        # plot points
        plt.scatter(point[0], point[1], c='r' if first else 'g')
        # plot word annotations
        plt.annotate(
            word, 
            xy = (point[0], point[1]),
            xytext = (-7, -6) if first else (7, -6),
            textcoords = 'offset points',
            ha = 'right' if first else 'left',
            va = 'bottom',
            size = "x-large"
        )
        first = not first if alternate else first

    # draw arrows
    if arrows:
        for i in range(0, len(words)-1, 2):
            a = vectors2d[i][0] + 0.04
            b = vectors2d[i][1]
            c = vectors2d[i+1][0] - 0.04
            d = vectors2d[i+1][1]
            plt.arrow(a, b, c-a, d-b,
                shape='full',
                lw=0.1,
                edgecolor='#bbbbbb',
                facecolor='#bbbbbb',
                length_includes_head=True,
                head_width=0.08,
                width=0.01
            )

    # draw diagram title
    if title:
        plt.title(title)

def build_neighbors(word, model, nviz=10):
    g = nx.Graph()
    g.add_node(word, color='r')
    viz1 = model.most_similar(word, topn=nviz)
    for v in viz1:
        g.add_node(v[0], color='b')
    g.add_weighted_edges_from([(word, v, w) for v,w in viz1 if w> 0.65])
    for v in viz1:
        for l in model.most_similar(v[0], topn=nviz):
            g.add_node(l[0], color='y')
        g.add_weighted_edges_from([(v[0], v2, w2) for v2,w2 in model.most_similar(v[0])])
    for v in viz1:
        g.add_node(v[0], color='b')
    g.add_node(word, color='r')
    return g

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    plt.tight_layout()
    #plt.savefig(f'./outputs/{title}')
    plt.show()

