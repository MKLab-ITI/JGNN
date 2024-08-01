# JGNN 

A native Java library for graph neural networks. 

## :dart: About

Graph Neural Networks (GNN) are getting more and more popular;
they can reason from relational information and perform inference from small datasets.
JGNN provides implementations that do not require dedicated hardware or firmware.

* Cross-platform (written in native java)
* Lightweight
* Minimal memory footprint
* Parallelized batching
* [Neuralang](tutorials/Neuralang.md) scripting language for model definition

## :rocket: Setup and links

Get the JAR file for a specific version from the project's releases
[releases](https://github.com/MKLab-ITI/JGNN/releases). Add this to your dependencies.
Alternatively, include the latest nightly version as a Maven or Gradle dependency 
by following the instructions of the JitPack distribution:

[![](https://jitpack.io/v/MKLab-ITI/JGNN.svg)](https://jitpack.io/#MKLab-ITI/JGNN)


:cyclone: [Tutorials](tutorials/README.md)<br>
:cyclone: [Javadoc](https://mklab-iti.github.io/JGNN/)


## :thumbsup: Contributing

Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/JGNN/issues). In addition to bug reports, requests for features and clarifications are welcome.
 

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
