# JGNN
A native Java library for Graph Neural Networks.

## Introduction
Graph Neural Networks (GNNs) have seen a dramatic increase in popularity
thanks to their ability to understand relations between nodes.
This library aims to provide GNN capabilities to native Java applications, 
for example to perform machine learning on Android. It does so by avoiding
the overhead of employing c- or python- based machine learning libraries,
such as TensorFlow Lite, that are often designed with pure performance in mind,
but from which GNNs do not get a large boost (operations such as graph convolutions
still  exhibit the big-O time complexity).


## Installation
For quick installation of the latest working version of the library, you can include it as Maven or Gradle dependency by following the instructions found in the following JiPack distribution page:
[![](https://jitpack.io/v/maniospas/jgnn.svg)](https://jitpack.io/#maniospas/jgnn)


## Examples
Under construction. Look at the package mklab.JGNN.examples for working examples. For this first draft version, only relational GNNs work correctly.

## Quick set up of a GNN
Under construction

## Tutorials
 
#### Primitives
[link](tutorials/Primitives.md) A collection of vector and matrix primitives support
low-level arithmetic operations, in-place version of the same operations and
both sparse and dense memory structures. The can also support applications
outside the scope of the library. This tutorial covers the use of machine learning
primitives.

#### Models and builders
[link](tutorials/Models.md) Learning tasks can be automated by defining symbolic
pipelines. This tutorial covers tools that can be used to create trainable
machine learning models, including builders that can build those tools from 
simple String expressions.

#### Datasets
A collection of publically available datasets that can be automatically
downloaded and imported from respective external sources to facilitate
benchmarking tasks. 