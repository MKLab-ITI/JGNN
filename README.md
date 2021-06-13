# JGNN
A native Java library for Graph Neural Networks.

## Introduction
Graph Neural Networks (GNNs) have seen a dramatic increase in popularity thanks to their ability to understand relations between nodes.
This library aims to provide GNN predictions to native Java applications, for example deployed on Android.
It does so by avoiding the overhead of employing other machine learning architectures, such as TensorFlow Lite, that are often designed
with pure performance in mind, which GNNs do not get a large boost out of (since operations such as graph convolutions still exhibit the
big-O time complexity).

This library helps bring popular GNN architectures to Java (e.g. Android) applications *including* the ability to train these architectures
on the fly as more data become accessible. For example, it can be used for Edge computing without needing to be aware of underlying hardware.


## Installation
For quick installation of the latest working version of the library, you can include it as Maven or Gradle dependency by following the instructions found in the following JiPack distribution page:
[![](https://jitpack.io/v/maniospas/jgnn.svg)](https://jitpack.io/#maniospas/jgnn)


## Examples
Under construction. Look at the package mklab.JGNN.examples for working examples. For this first draft version, only relational GNNs work correctly.

## Quick set up of a graph neural network
Under construction

## Tutorials
 
**Primitives.** A collection of vector and matrix primitives support low-level arithmetic operations, in-place version of the same operations and both sparse and dense memory structures. The can also support applications outside the scope of the library. For details on primitives please visit [this link](docs/Primitives.md)

**Models and builders.** Parameter learning tasks can be automated by defining symbolic pipelines. To this end, builder tools can be used to create models from String expressions. For details please visit [this link](docs/Models.md).

**Datasets.** A collection of publically available datasets that can be automatically downloaded and imported from respective external sources to facilitate benchmarking tasks.