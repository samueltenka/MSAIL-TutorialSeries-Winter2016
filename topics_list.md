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
Aaron and Samuel will

### Who is the Tutorial Series for?

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
I also generated these topics by looking online, trying to maximize
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
0           4        7  8       11           familiar enough to explain
   1     3     5 N/A               12        familiar, need to review
      2                    9 10       13 14  not familiar at all
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
	6.   <Buffer in case 5. is too long, etc.>
	7.   Feature-Finding. *Autoencoders*. Neural Net Auto-encoders.
	8.   Feature-Finding. Restricted Boltzmann Machines.
	9.   Feature-Finding. Deep Belief Networks.
	10.  Time Series. Kalman Filter. *Belief Propagation*.
	11.  Time Series/Graphical Models: Hidden Markov Models
	12.  Graphical Models: Inference and Sampling.
	13.  Graphical Models: Learning Structure
	14.  Neat Example: Grammar Induction

Especially general and important techniques are in *italics*.

Locality Sensitive Hashing
Heirarchical clustering.

### Resources
* A nice overview of unsupervised learning:
http://mlg.eng.cam.ac.uk/zoubin/papers/ul.pdf

* Dirichlet Processes tutorial (Slides):
http://www.cs.cmu.edu/~kbe/dp_tutorial.pdf

* Kernel Trick and Nonlinear Dimension Reduction (blog w/ code):
http://sebastianraschka.com/Articles/2014_kernel_pca.html

* Learning Graphical Model Structure (Slides)
http://spark-university.s3.amazonaws.com/stanford-pgm/slides/Section-5-Learning-BN-Structures.pdf

* Graphical Models, Bayes Nets:
http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html

* Grammar Induction (Survey Paper):
http://jmlr.org/papers/volume12/glowacka11a/glowacka11a.pdf

* Convolutional Neural Nets for Facial Keypoint labeling tutorial (blog w/ code)
(**not unsupervised** but really awesome, and comes with code, dataset, etc.):
http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
