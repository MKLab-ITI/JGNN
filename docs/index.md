# JGNN

Graph Neural Networks (GNNs) are getting more and more popular as a machine learning paradigm, for example to make predictions based on relational information, or to perform inference on small datasets. JGNN is a library that provides cross-platform implementations of this paradigm without the need for dedicated hardware or firmware; create highly portable models that fit and are trained in a few megabytes of memory. Find GNN builders, training strategies, and datasets for out-of-the-box experimentation.

While reading this guide, keep in mind that this is not a library for running computationally intensive stuff; it has no GPU support and we do not plan to add any (unless such support becomes integrated in the Java virtual machine). So, while source code is highly optimized and complex architectures are supported, running them quickly on graphs with many nodes may require compromises in the number of learned parameters or running time.

