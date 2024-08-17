# JGNN 

A native Java library for graph neural networks.

**Current requirements: Java 17 or later**

## :dart: About

Graph Neural Networks (GNNs) are getting more and more popular, for example to
make predictions based on relational information, and to perform inference
on small datasets. JGNN provides cross-platform implementations of this machine
learning paradigm that do not require dedicated hardware or firmware.

* Cross-platform
* Lightweight
* Optimized: data views, automatic datatypes, SIMD, parallelized batching
* [Neuralang](tutorials/Neuralang.md) scripting language for model definition

## :rocket: Setup and links

Add to your dependencies the JAR file of a specific version. Download this
from the project's [releases](https://github.com/MKLab-ITI/JGNN/releases). .
Alternatively, include the latest nightly version as a Maven or Gradle dependency 
by following the instructions of the JitPack distribution:

[![](https://jitpack.io/v/MKLab-ITI/JGNN.svg)](https://jitpack.io/#MKLab-ITI/JGNN)


:cyclone: [Tutorials](tutorials/README.md)<br>
:cyclone: [Javadoc](https://mklab-iti.github.io/JGNN/)


## :notebook: Citation

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

<details>
<summary> <b>Changes since the publication's v1.0.0</b> </summary>

* Introduced [Neuralang](tutorials/Neuralang.md)
* Autosized parameteters
* Up to 30% less memory 
* Up to 80% less running time
* Renamed `GCNBuilder` to `FastBuilder`
* Neighbor attention and message passing
* Sort pooling and graph classification

</details>


## :thumbsup: Contributing

Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/JGNN/issues). In addition to bug reports, 
requests for features and clarifications are welcome.

**Copyright Emmanouil Krasanakis (maniospas@hotmail.com), Apache license 2.0** 
 