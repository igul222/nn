import re
import numpy as np

INTERVAL = 10000
LABELS = ['secs/iter', 'train cost', 'train kl1', 'train kl2']

for label in LABELS:
    print "=============================="
    print label

    p = re.compile(label+':[0-9]+\.[0-9]+')

    vals = []

    with open('output.txt') as f:
        for line in f:
            vals.extend(p.findall(line))

    vals = [float(x[len(label)+1:]) for x in vals]


    for i in xrange(0, len(vals), INTERVAL):
        print "{}-{}\t{}".format(i, min(len(vals), i+INTERVAL), np.mean(vals[i:i+INTERVAL]))