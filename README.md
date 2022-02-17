# JGNN
A native Java library for Graph Neural Networks.

# :brain: About
Graph Neural Networks (GNNs) have seen a dramatic increase in popularity
thanks to their ability to understand relations between nodes.
This library aims to provide GNN capabilities to native Java applications, 
for example to perform machine learning on Android. It does so by avoiding
c-based machine learning libraries, such as [TensorFlow Lite](https://www.tensorflow.org/lite),
that are often designed with pure performance in mind but which often require
specific hardware to run, such as GPUs, and drastically increase the size of
deployed applications.


# :hammer_and_wrench: Installation
For quick installation of the latest working version of the library, you can include it as Maven or Gradle dependency by following the instructions found in the following JiPack distribution page:
[![](https://jitpack.io/v/maniospas/jgnn.svg)](https://jitpack.io/#maniospas/jgnn)

# :zap: Quickstart
1. [Introduction to JGNN](tutorials/Introduction.md)

# :fire: Features
* Cross-platform support (written in native java)
* Reduced memory consumption and allocation speed ups
* Minimal impact to application size

# :link: Material
* [Javadoc](https://maniospas.github.io/JGNN/)
* [Introduction to JGNN primitive](tutorials/Primitives.md) A collection of vector and matrix primitives support low-level arithmetic operations, in-place version of the same operations and both sparse and dense memory structures. The can also support applications outside the scope of the library. This tutorial covers the use of machine learning primitives.
* [Introduction to JGNN models and builders](tutorials/Models.md) Learning tasks can be automated by defining symbolic pipelines. This tutorial covers tools that can be used to create trainable machine learning models, including builders that can build those tools from  simple String expressions.
* [Introduction to JGNN benchmarking] A collection of publically available datasets that can be automatically downloaded and imported from respective external sources to facilitate benchmarking tasks. 

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker]() or by participating in [discussions]().
Please check out the [contribution guidelines](CONTRIBUTING.md) to bring modifications to the code base.
 
# :notebook: Citation
TBD