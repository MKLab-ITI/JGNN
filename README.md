# JGNN

A native Java library for Graph Neural Networks.

# :dart: About

Graph Neural Networks (GNNs) are getting more and more popular;
they can reason from relational information and perform inference from small datasets.
JGNN implements GNNs for native Java applications to supports cross-platform machine
learning, such as on Android, without dedicated hardware or firmware.

* [Tutorials](tutorials/README.md)
* [Javadoc](https://mklab-iti.github.io/JGNN/)

# :rocket: Setup and features

Find the latest nightly release as as JAR from [here](https://github.com/MKLab-ITI/JGNN/releases/latest/download/JGNN.jar).
Download this and add it in your project's dependencies.
To install the latest stable version, include it as a Maven or Gradle dependency 
by following the instructions of its JitPack distribution:
[![](https://jitpack.io/v/MKLab-ITI/JGNN.svg)](https://jitpack.io/#MKLab-ITI/JGNN)

* Cross-platform (written in native java)
* Lightweight
* Minimal memory footprint
* Parallelized batching
* [Neuralang](tutorials/Neuralang.md) scripting language for model definition


# :thumbsup: Contributing

Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/JGNN/issues). In addition to bug reports, requests for features and clarifications are welcome. Please check out the [contribution guidelines](CONTRIBUTING.md) when bringing modifications to the code base.
 


# :notebook: Citation

```
@article{krasanakis2023101459,
	title = {JGNN: Graph Neural Networks on native Java},
	journal = {SoftwareX},
	volume = {23},
	pages = {101459},
	year = {2023},
	issn = {2352-7110},
	doi = {https://doi.org/10.1016/j.softx.2023.101459},
	url = {https://www.sciencedirect.com/science/article/pii/S2352711023001553},
	author = {Emmanouil Krasanakis and Symeon Papadopoulos and Ioannis Kompatsiaris}
}
```

# :cyclone: Changes since 1.0.0

* [Neuralang](tutorials/Neuralang.md)
* Autosized parameteters
* Up to 30% less memory 
* Up to 90% less running time
* Renamed `GCNBuilder` to `FastBuilder`
* Neighbor attention and message passing
* Sort pooling and graph classification