import torch
import numpy as np

SQRT_CONST = 1e-10

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, np.inf))

def lindisc(X,p,t,l):
    ''' Linear MMD '''

    it = torch.where(t==l)
    ic = torch.where(t!=l)

    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.reduce_mean(Xc,reduction_indices=0)
    mean_treated = torch.reduce_mean(Xt,reduction_indices=0)

    c = torch.square(2*p-1)*0.25
    f = torch.sign(p-0.5)

    mmd = torch.reduce_sum(torch.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p,l):
    ''' Linear MMD '''

    it = torch.where(t==l)
    ic = torch.where(t!=l)

    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.reduce_mean(Xc,reduction_indices=0)
    mean_treated = torch.reduce_mean(Xt,reduction_indices=0)

    mmd = torch.reduce_sum(torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,l,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = torch.where(t==l)
    ic = torch.where(t!=l)

    Xc = X[ic]
    Xt = X[it]

    Kcc = torch.exp(-pdist2sq(Xc,Xc)/torch.square(sig))
    Kct = torch.exp(-pdist2sq(Xc,Xt)/torch.square(sig))
    Ktt = torch.exp(-pdist2sq(Xt,Xt)/torch.square(sig))

    m = torch.to_float(Xc.size()[0])
    n = torch.to_float(Xt.size()[0])

    mmd = torch.square(1.0-p)/(m*(m-1.0))*(torch.reduce_sum(Kcc)-m)
    mmd = mmd + torch.square(p)/(n*(n-1.0))*(torch.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*torch.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*torch.matmul(X,torch.transpose(Y, 0, 1))
    nx = torch.sum(torch.square(X), 1, keepdim=True)
    ny = torch.sum(torch.square(Y), 1, keepdim=True)
    D = (C + torch.transpose(ny, 0, 1)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t,l):
    it = torch.where(t==l)
    ic = torch.where(t!=l)
    Xc = X[ic]
    Xt = X[it]
    # nc = torch.to_float(Xc.size()[0])
    # nt = torch.to_float(Xt.size()[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,l,lam=10,its=10,sq=False,backpropT=False,device='cpu'):
    """ Returns the Wasserstein distance between treatment groups """

    # print("p: ", p)

    it = torch.where(t==l)
    ic = torch.where(t!=l)
    # print("it: ", it)
    # print("ic: ", ic)
    Xc = X[ic]
    Xt = X[it]
    # print("Xc: ", Xc)
    # print("Xt: ", Xt)
    nc = Xc.size()[0]
    nt = Xt.size()[0]
    # print("nc: ", nc)
    # print("nt: ", nt)

    if nc == 0 or nt == 0:
        return 0.0, 0.0

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))
    
    # print("M: ", M)

    ''' Estimate lambda and delta '''
    M_mean = torch.mean(M)
    # M_drop = torch.nn.dropout(M,10/(nc*nt))
    delta = torch.max(M).detach()
    eff_lam = (lam/M_mean).detach()

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*torch.ones_like(M[0:1,:], device=device)
    col = torch.cat([delta*torch.ones_like(M[:,0:1], device=device),torch.zeros((1,1), device=device)], 0)
    Mt = torch.cat([M,row], 0)
    Mt = torch.cat([Mt,col], 1)

    ''' Compute marginal vectors '''
    a = torch.cat((p*torch.ones((nt, 1), device=device)/nt, (1-p)*torch.ones((1,1), device=device)), 0)
    b = torch.cat(((1-p)*torch.ones((nc, 1), device=device)/nc, p*torch.ones((1,1), device=device)), 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = torch.exp(-Mlam) + 1e-6 # added constant to avoid nan
    # U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(torch.matmul(ainvK,(b/torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))))
    v = b/(torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))

    T = u*(torch.transpose(v, 0, 1)*K)

    if not backpropT:
        T = T.detach()

    E = T*Mt
    D = 2*torch.sum(E)

    # print("D: ", D)

    return D, Mlam

def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.size()[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w
