import numpy as np
import scipy.io
import sklearn.neighbors

np.random.seed(1234)

train_data = scipy.io.loadmat('/home/ishaan/data/svhn/train_32x32.mat')
test_data = scipy.io.loadmat('/home/ishaan/data/svhn/test_32x32.mat')
extra_data = scipy.io.loadmat('/home/ishaan/data/svhn/extra_32x32.mat')

train_data_len = len(train_data['y'])

train_11K_indices = np.random.choice(train_data_len, size=11000, replace=False)
train_1K_indices, train_10K_indices = train_11K_indices[:1000], train_11K_indices[1000:]

train_X = train_data['X'].transpose(3,2,0,1)[train_1K_indices]
train_y = train_data['y'][train_1K_indices]

val_X = train_data['X'].transpose(3,2,0,1)[train_10K_indices]
val_y = train_data['y'][train_10K_indices]

test_X = test_data['X'].transpose(3,2,0,1)
test_y = test_data['y']

train_y = train_y.flatten()
val_y = val_y.flatten()
test_y = test_y.flatten()

# raw pixels
def raw_extract_feats(x):
    return x.reshape((-1, 3072))

# random features
W = np.random.normal(size=(32*32*3, 256))
def random_proj_extract_feats(x):
    return np.dot(x.reshape((-1, 3072)), W)

def batch_extract_feats(X, extract_feats):
    results = []
    for i in xrange(1+(len(X)/128)):
        results.append(extract_feats(X[128*i:128*(i+1)]))
    return np.concatenate(results, axis=0)

def run_knn(extract_feats):

    _train_X = batch_extract_feats(train_X, extract_feats)
    _val_X = batch_extract_feats(val_X, extract_feats)
    # test_X = batch_extract_feats(train_X, extract_feats)

    clf = sklearn.neighbors.KNeighborsClassifier(1, weights='uniform')
    clf.fit(_train_X, train_y)
    val_error = 1. - np.mean(np.equal(val_y, clf.predict(_val_X)))
    print "{} val".format(val_error)
    # test_error = 1. - np.mean(np.equal(test_y, clf.predict(test_X)))
    # print "{} val\t{} test".format(val_error, test_error)
    return val_error

if __name__ == '__main__':
    run_knn(random_proj_extract_feats)
