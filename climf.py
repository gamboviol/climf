"""
CLiMF Collaborative Less-is-More Filtering, a variant of latent factor CF
which optimises a lower bound of the smoothed reciprocal rank of "relevant"
items in ranked recommendation lists.  The intention is to promote diversity
as well as accuracy in the recommendations.  The method assumes binary
relevance data, as for example in friendship or follow relationships.

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
ACM RecSys 2012
"""

from math import exp, log
import numpy as np

def g(x):
    """sigmoid function"""
    return 1/(1+exp(-x))

def dg(x):
    """derivative of sigmoid function"""
    return exp(x)/(1+exp(x))**2

def precompute_f(data,U,V,i):
    """precompute f[j] = <U[i],V[j]>
    params:
      data: scipy csr sparse matrix containing user->(item,count)
      U   : user factors
      V   : item factors
      i   : item of interest
    returns:
      dot products <U[i],V[j]> for all j in data[i]
    """
    items = data[i].indices
    f = dict((j,np.dot(U[i],V[j])) for j in items)
    return f

def objective(data,U,V,lbda):
    """compute objective function F(U,V)
    params:
      data: scipy csr sparse matrix containing user->(item,count)
      U   : user factors
      V   : item factors
      lbda: regularization constant lambda
    returns:
      current value of F(U,V)
    """
    F = -0.5*lbda*(np.sum(U*U)+np.sum(V*V))
    for i in xrange(len(U)):
        f = precompute_f(data,U,V,i)
        for j in f:
            F += log(g(f[j]))
            for k in f:
                F += log(1-g(f[k]-f[j]))
    return F

def update(data,U,V,lbda,gamma):
    """update user/item factors using stochastic gradient ascent
    params:
      data : scipy csr sparse matrix containing user->(item,count)
      U    : user factors
      V    : item factors
      lbda : regularization constant lambda
      gamma: learning rate
    """
    for i in xrange(len(U)):
        dU = -lbda*U[i]
        f = precompute_f(data,U,V,i)
        for j in f:
            dV = g(-f[j])-lbda*V[j]
            for k in f:
                dV += dg(f[j]-f[k])*(1/(1-g(f[k]-f[j]))-1/(1-g(f[j]-f[k])))*U[i]
            V[j] += gamma*dV
            dU += g(-f[j])*V[j]
            for k in f:
                dU += (V[j]-V[k])*dg(f[k]-f[j])/(1-g(f[k]-f[j]))
        U[i] += gamma*dU

def compute_mrr(data,U,V,test_users=None):
    """compute average Mean Reciprocal Rank of data according to factors
    params:
      data      : scipy csr sparse matrix containing user->(item,count)
      U         : user factors
      V         : item factors
      test_users: optional subset of users over which to compute MRR
    returns:
      the mean MRR over all users in data
    """
    mrr = []
    if test_users is None:
        test_users = range(len(U))
    for ix,i in enumerate(test_users):
        items = set(data[i].indices)
        predictions = np.sum(np.tile(U[i],(len(V),1))*V,axis=1)
        for rank,item in enumerate(np.argsort(predictions)[::-1]):
            if item in items:
                mrr.append(1.0/(rank+1))
                break
    assert(len(mrr) == len(test_users))
    return np.mean(mrr)

if __name__=='__main__':

    from optparse import OptionParser
    from scipy.io.mmio import mmread
    import random

    parser = OptionParser()
    parser.add_option('--train',dest='train',help='training dataset (matrixmarket format)')
    parser.add_option('--test',dest='test',help='optional test dataset (matrixmarket format)')
    parser.add_option('-d','--dim',dest='D',type='int',default=10,help='dimensionality of factors (default: %default)')
    parser.add_option('-l','--lambda',dest='lbda',type='float',default=0.001,help='regularization constant lambda (default: %default)')
    parser.add_option('-g','--gamma',dest='gamma',type='float',default=0.0001,help='gradient ascent learning rate gamma (default: %default)')
    parser.add_option('--max_iters',dest='max_iters',type='int',default=25,help='max iterations (default: %default)')

    (opts,args) = parser.parse_args()
    if not opts.train or not opts.D or not opts.lbda or not opts.gamma:
        parser.print_help()
        raise SystemExit

    data = mmread(opts.train).tocsr()  # this converts a 1-indexed file to a 0-indexed sparse array
    if opts.test:
        testdata = mmread(opts.test).tocsr()

    U = 0.01*np.random.random_sample((data.shape[0],opts.D))
    V = 0.01*np.random.random_sample((data.shape[1],opts.D))

    num_train_sample_users = min(data.shape[0],1000)
    train_sample_users = random.sample(xrange(data.shape[0]),num_train_sample_users)
    print 'train mrr = {0:.4f}'.format(compute_mrr(data,U,V,train_sample_users))
    if opts.test:
        num_test_sample_users = min(testdata.shape[0],1000)
        test_sample_users = random.sample(xrange(testdata.shape[0]),num_test_sample_users)
        print 'test mrr  = {0:.4f}'.format(compute_mrr(testdata,U,V,test_sample_users))

    for iter in xrange(opts.max_iters):
        update(data,U,V,opts.lbda,opts.gamma)
        print 'iteration {0}:'.format(iter+1)
        print 'objective = {0:.4f}'.format(objective(data,U,V,opts.lbda))
        print 'train mrr = {0:.4f}'.format(compute_mrr(data,U,V,train_sample_users))
        if opts.test:
            print 'test mrr  = {0:.4f}'.format(compute_mrr(testdata,U,V,test_sample_users))

    print 'U',U
    print 'V',V
