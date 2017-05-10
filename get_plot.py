#!/usr/bin/env python

import argparse
import cPickle as pickle
import subprocess
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--ema', type=float, default=0.)
parser.add_argument('--xlim', type=str, default='')
parser.add_argument('--ylim', type=str, default='')
parser.add_argument('--cached', action='store_true')
parser.add_argument('series', nargs='*')
args = parser.parse_args()
for series_str in args.series:
	host, experiment, series = series_str.split(':')

	if not args.cached:
		print "{}: cpr".format(series_str)
		ls = subprocess.check_output("ssh  {} 'cp experiments/{}/log.pkl /tmp/log_{}.pkl'".format(host, experiment, series_str), shell=True)
		print "{}: downloading".format(series_str)
		subprocess.check_output("scp {}:/tmp/log_{}.pkl /tmp/log_{}.pkl".format(host, series_str, series_str), shell=True)

	with open('/tmp/log_{}.pkl'.format(series_str), 'rb') as f:
		data = pickle.load(f)

	if series in data:
		data = data[series]
	else:
		data = data[series.replace('_',' ')]

	x_vals = np.sort(data.keys())
	y_vals = [data[x] for x in x_vals]

	y_vals_ema = [y_vals[0]]
	for i in xrange(1, len(y_vals)):
		y_vals_ema.append((args.ema * y_vals_ema[-1]) + ((1.-args.ema)*y_vals[i]))

	plt.plot(x_vals, y_vals_ema, label=series_str)

plt.xlabel('iteration')
if len(args.xlim)>0:
	x_lower, x_upper = [float(x) for x in args.xlim.split(',')]
	plt.xlim(x_lower, x_upper)
if len(args.ylim)>0:
	y_lower, y_upper = [float(x) for x in args.ylim.split(',')]
	plt.ylim(y_lower, y_upper)
plt.legend()
plot_path = '/tmp/plot_{}.jpg'.format(str(time.time()))
plt.savefig(plot_path)
subprocess.check_output('open "{}"'.format(plot_path), shell=True)