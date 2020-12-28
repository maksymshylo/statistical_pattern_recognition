from scipy.stats import multivariate_normal
import numpy as np

def predict_proba(X,n_components,mu,sigma,phi):
    n,_ = X.shape
    likelihood = np.zeros( (n, n_components) )
    for i in range(n_components):
        distribution = multivariate_normal(mean=mu[i], cov=sigma[i])
        likelihood[:,i] = distribution.pdf(X)
    
    numerator = likelihood * phi
    weights = numerator / np.sum(numerator,axis=1)[:, np.newaxis]
    return weights

def em(X,n_components,mu,sigma,weights,phi):
    # expectation step
    weights =  predict_proba(X,n_components,mu,sigma,phi)
    phi = np.mean(weights,axis=0)

    # maximization step
    for i in range(n_components):
        weight =  weights[:, [i]]
        total_weight = weight.sum()
        mu[i] = np.sum(X * weight,axis=0) / total_weight
        sigma[i] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)
    return mu,sigma,weights,phi

def gmm(X,n_components, n_iter = 100):

    # initialization
    n,m = X.shape
    phi = np.full(n_components, 1/n_components)
    weights = np.full(X.shape,  1/n_components)
    random_row = np.random.randint(low=0, high=n, size=n_components)
    mu = list(X[random_row,:])
    sigma = [np.cov(X.T)]*n_components

    # em
    for _ in range(n_iter):
        mu,sigma,weights,phi = em(X,n_components,mu,sigma,weights,phi)

    return mu,sigma,weights,phi