Topics for MSAIL Tutorial Series, Winter 2016.
==============================================

Summary
-------
### What is Unsupervised Learning?
This semester we'll focus on **unsupervised learning**, that is,
the automatic finding of structure in un-labelled datasets. While
supervised learning algorithms fill in missing coordinate values,
unsupervised learning finds a useful coordinate system, that is,
one in which the dataset's coordinate values becomes sparse and independent.
Thus, cluster-detection, principal component analysis, dimension-reduction,
and feature-learning in general are problems of unsupervised learning.
In other words, while supervised learning predicts labels *given features*,
unsupervised learning *learns features*. Unsupervised hence helps supervised.

### What, When, and Where will be the MSAIL Tutorial Series, Winter 2016?
Aaron and Samuel will...
**Question**
Hmm, yeah, so is Wednesday 7-8pm finalized? (would work for Sam.)
Is 50 minutes optimal? I'd guess a bit longer would be better, but I don't know.
*(end of Question)*

### Who is the Tutorial Series for?
**Question**
Who's our target audience? How much do they know, and what do they want to see?
*(end of Question)*

### Resources

Content
-------
### Topics Outline
**Question**.
Aaron, what do you think of this plan? I'm open to any plan.

This is the 0th iteration of what I hope will be a convergent process;
right now it probably needs lots of modification. For example, you mentioned the
tutorial members are hoping for unsupervised and advanced supervised learning,
but the topics below are all unsupervised (suggestions for supervised topics?).
Also, I generated these topics by looking online, trying to maximize
well-roundedness in the world of unsupervised learning; but really, we want to
account for how well we know the topics and the quantity and quality of extant
resources (e.g iPython notebooks).

Specifically, the topics sweep through 4 unsupervised learning problems:
Clustering, Dimension Reduction, Deep Learning of Features, and Learning on
Graphical Models.
Are these the important topics? Are the topics ordered in a sensible way?
Is it too much, too little?

I might be missing some big part of unsupervised learning. For example, do
we want to discuss Anomaly Detection, Density Estimation, Generative models?
I'm having trouble characterizing unsupervised learning more precisely than
"Anything we can learn from with unlabeled data"; a precise characterization
would be useful to understand which topics are important and why, and how we
should arrange them.

I'm not an expert on any topic; I'd slice them up this way:

	0           4     7     8       11           familiar enough to explain
	   1     3     5     6             12        familiar, need to review
	      2                    9 10       13 14  not familiar at all

Also, how are unsupervised models validated?
*(end of Question)*

We'll meet once a week for 15 weeks. Tentative topics:

	0.   What is Unsupervised Learning?
	     Clustering. Problem & Example.
	1.   Clustering. k-means. *Expectation-Maximization*. Gaussian Mixtures.
	2.   Clustering. Dirichlet Processes. *Markov Chain Monte Carlo*.
	3.   Neat Example: A Recommender System (as in Ng's Coursera course)
	4.   Dimension Reduction. Principal and Independent Component Analyses.
	5.   Dimension Reduction: Nonlinear. *Kernel Trick*. Kernel PCA.
	     Isomap. Locally Linear Embedding. Self-Organizing Maps.
	6.   Feature-Finding. *Autoencoders*. Neural Net Auto-encoders.
	7.   Neat Example: Word2Vec.
	8.   Feature-Finding. Restricted Boltzmann Machines.
	9.   Feature-Finding. Deep Belief Networks.
	10.  Time Series. Kalman Filter. *Belief Propagation*.
	11.  Time Series/Graphical Models: Hidden Markov Models
	12.  Graphical Models: Inference and Sampling.
	13.  Graphical Models: Learning Structure
	14.  Neat Example: Grammar Induction

Especially general and important techniques are in *italics*.

Locality Sensitive Hashing, Heirarchical clustering.

### Resources: 00 Intro; K-Means
* A nice overview of unsupervised learning:
http://mlg.eng.cam.ac.uk/zoubin/papers/ul.pdf
* k-Means (iPython notebook):
https://github.com/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-k-means.ipynb
### Resources: 01 Expectation Maximization. K-Means. Gaussian Mixtures.
* k-Means and PCA (iPython notebook):
https://github.com/jdwittenauer/ipython-notebooks/blob/master/ML-Exercise7.ipynb
* Gaussian Mixture Models (iPython notebook)
https://github.com/donnemartin/data-science-ipython-notebooks/blob/master/scikit-learn/scikit-learn-gmm.ipynb
### Resources: 02 Dirichlet
* Dirichlet Processes tutorial (Slides):
http://www.cs.cmu.edu/~kbe/dp_tutorial.pdf
* Direchlet Processes (iPython notebooks):
https://github.com/tdhopper/notes-on-dirichlet-processes
and for application to clustering algorithms, see especially
https://github.com/tdhopper/notes-on-dirichlet-processes/blob/master/2015-09-02-fitting-a-mixture-model.ipynb

### Resources: 03 Recommender Systems
* Anomaly Detection and Recommender Systems (iPython notebook):
https://github.com/jdwittenauer/ipython-notebooks/blob/master/ML-Exercise8.ipynb
### Resources: 04 PCA and ICA
* k-Means and PCA (iPython notebook):
https://github.com/jdwittenauer/ipython-notebooks/blob/master/ML-Exercise7.ipynb
* Spark PCA (iPython notebook):
http://nbviewer.ipython.org/github/jdwittenauer/ipython-notebooks/blob/master/Spark-ML-Lab5-NeuroPCA.ipynb
### Resources: 05 Clustering
* Kernel Trick and Nonlinear Dimension Reduction (blog w/ iPython notebook):
Blog:
http://sebastianraschka.com/Articles/2014_kernel_pca.html
iPython Notebook:
http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/kernel_pca.ipynb

### Resources: 06 Autoencoders
### Resources: 07 Word2Vec
* Word2Vec (iPython notebook):
https://github.com/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-exercises/5_word2vec.ipynb
* Language Exploration using Vector Space Models (iPython notebook):
http://nbviewer.ipython.org/github/jdwittenauer/ipython-notebooks/blob/master/LanguageVectors.ipynb
### Resources: 08 Restricted Boltzmann Machines
### Resources: 09 Deep Belief Networks
* Deep Dream (iPython notebook):
https://github.com/donnemartin/data-science-ipython-notebooks/tree/master/deep-learning/deep-dream

### Resources: 10 Clustering
### Resources: 11 Clustering
### Resources: 12 Clustering
### Resources: 13 Clustering
### Resources: 14 Clustering

* Learning Graphical Model Structure (Slides)
http://spark-university.s3.amazonaws.com/stanford-pgm/slides/Section-5-Learning-BN-Structures.pdf
* Graphical Models, Bayes Nets:
http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
* Grammar Induction (Survey Paper):
http://jmlr.org/papers/volume12/glowacka11a/glowacka11a.pdf

* Convolutional Neural Nets for Facial Keypoint labeling tutorial (blog w/ code)
(**not unsupervised** but really awesome, and comes with code, dataset, etc.):
http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
