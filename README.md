# JGNN
A native Java library for Graph Neural Networks.

# :brain: About
Graph Neural Networks (GNNs) have seen a dramatic increase in popularity
thanks to their ability to understand relations between graph nodes.
This library aims to provide GNN capabilities to native Java applications, 
for example to perform machine learning on Android. It does so by avoiding
c-based machine learning libraries, such as [TensorFlow Lite](https://www.tensorflow.org/lite).
The latter are often designed with pure performance in mind but often require
specific hardware to run, such as GPUs, drastically increase the size of
deployed applications.


# :hammer_and_wrench: Installation
For quick installation of the latest working version of the library, you can include it as Maven or Gradle dependency by following the instructions found in the following JiPack distribution page:
[![](https://jitpack.io/v/maniospas/jgnn.svg)](https://jitpack.io/#maniospas/jgnn)

# :zap: Quickstart
1. [Introduction to JGNN](tutorials/Introduction.md)

# :fire: Features
* Cross-platform support (written in native java)
* Lightweight
* Speed ups and minimal memory footprint with direct memory access
* Parallelized batching

# :link: Material
* [Javadoc](https://maniospas.github.io/JGNN/)
* [Primitives](tutorials/Primitives.md)
* [Models and symbolic builders](tutorials/Models.md)
* [Benchmarking]

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/JGNN/issues).
Please check out the [contribution guidelines](CONTRIBUTING.md) to bring modifications to thecode base.
 
# :notebook: Citation
TBD