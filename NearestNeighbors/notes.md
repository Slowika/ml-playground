# Sklearn doc

Main idea: find a predefined number of training samples closest in distance to the new point,
and predict the label from these

Unsupervised NN: manifold learning and spectral clustering
Supervised NN: class and reg

K-Nearest Neighbor Learning and Radius-based Neighbor Learning

Classification:
- `KNeighborsClassifier` or `RadiusNeighborsClassifier` (might be better for a sparse data but the course of dimen)
- we can add weights for the neighbors based on eg. the distance

example of a :
    non-generalizing machine learning method (remember all the data)
    non-parametric model (good for irregular decision boundary - and data of unknown distribution)

Sparse matrices -> Minkowski metrics
(Euclidean metric: [1 0 0; 0 1 0; 0 0 1], Minkowski: [-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])

Kernel density estimation relies on NN (not really according to Bishop)

# Bishop

Parametric - functions with parameters determined by the dataset
Limitation: chosen density might be a poor model of the distribution which generates the data

Non-param require fewer assumptions about the data and work better when the true distribution is unknown/can't be estimated
(stackexchange)

Frequentist non-param methods: Kernel density estimator and NN

The problem with kernel density estimator: kernel parameter is fixed for all the kernels, over-smoothing or noise
The choice of kernel param should be dependent on data location => NN

Fixed K, V based on data
K - number of points within some region R
V - volume of R

K governs the degree of smoothing (too small: noise, too big: over-smooth)

So far it was about NN density estimation which can be applied to classification by:
    applying NN de to every class and using Bayes theorem

As a result: `p(C_k | x) = K_k / K`

C_k - a class
x - a point around which we make a sphere with K points and data-dependent radius
K - all points in the sphere
K_k - points which are in the sphere and in the class C_k

K = 1 : "nearest neighbor rule"

fun fact: for K = 1 and for N -> inf (N- total nr of data points) the error rate is never more than twice
the minimum achievable error rate of an optimal classifier, i.e., one that uses the true class distributions

Limiation of KNN: all data has to be stored (might be stored in a tree)
param are restricted by distributions
we want to find a model which is both flexible in terms of dist and which complexity is independent of the dataset
how? the next chapters

# Introduction to Machine Learning by Alex Smola

```# TODO```

# The Elements of Statistical Learning

```# TODO```

[https://ashokharnal.wordpress.com/tag/kneighborsclassifier-explained/](https://ashokharnal.wordpress.com/tag/kneighborsclassifier-explained/)
