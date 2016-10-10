import json
import numpy as np

INTERVAL = 10000
LABELS = ['train cost', 'train kl1', 'train kl2']

for label in LABELS:
    print "=============================="
    print label

    vals = []

    with open('train_output.ndjson') as f:
        for line in f:
            line = json.loads(line[:-1])
            vals.extend(line[label])

    for i in xrange(0, len(vals), INTERVAL):
        print "{}-{}\t{}".format(i, min(len(vals), i+INTERVAL), np.mean(vals[i:i+INTERVAL]))