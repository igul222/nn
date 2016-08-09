import numpy as np
import sklearn.decomposition.pca as skpca

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

datapath = '/data/lisatmp4/faruk/pixelvae/mnist_vae_pca/'
plotpath = '/data/lisatmp4/faruk/pixelvae/pca_plot_vae/'

files = [f for f in listdir(datapath) if isfile(join(datapath, f))]
names = []

for f in files:
    if f[:5] == 'iters':    names.append(f[5:])

for name in names:
    itername = 'iters' + name
    labelname = 'labels_iters' + name

    iter_ = int(itername.split('_')[0].split('iters')[-1])
    if iter_ == 300000 :
        mus = np.load(open(datapath + itername, 'r'))
        targets = np.load(open(datapath + labelname, 'r'))

        pca = skpca.PCA(n_components = 2)
        pca_mus = pca.fit_transform(mus)

        plt.scatter(pca_mus[:, 0], pca_mus[:, 1], c = targets, marker = '+')
        plt.savefig(plotpath + '{}.png'.format(iter_))
