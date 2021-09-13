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


# :hammer_and_wrench: Installation
For quick installation of the latest working version of the library, you can include it as Maven or Gradle dependency by following the instructions found in the following JiPack distribution page:
[![](https://jitpack.io/v/maniospas/jgnn.svg)](https://jitpack.io/#maniospas/jgnn)


# :zap: Quickstart
### Dataset
As an example, we will create a custom model to make predictions on the Lymphography dataset.
First, we load the dataset (it is automatically downloaded) and use the `IdConverter` class
to transform its labels and features into matrices. To save memory space, the library implicitly
makes these matrices sparse, although this is not explicitly defined. These matrices could also
be programmatically generated in either sparse or dense form (according to memory vs speed
considerations) using primitive operations.

Loaded features and labels are organized into `Matrix` instances. Originally, `IdConverter.oneHot`
constructs martices whose rows correspond to different nodes. However, inputting these in our
specific algorithm definition requires a transposition, so that columns are made to correspond
to nodes instead. To do this fastly, we use the `Matrix.asTransposed` method, which creates
a see-through interface that directly accesses the original matrix elements without allocating 
them anew.


```java
Dataset dataset = new Datasets.Lymphography();
IdConverter nodeIdConverter = dataset.nodes();
Matrix labels = nodeIdConverter.oneHot(dataset.labels()).asTransposed();
Matrix features = nodeIdConverter.oneHot(dataset.features()).asTransposed();
```

### Model definition
We then store the number of features and class labels that will help us define our model. It
is worth noting that matrix dimensions and elements are `long` numbers. Dense matrices can only
store up to integers, but this conversion allows us to handle large sparce matrices through the
same interfaces.

```java
long numFeatures = features.getRows();
long numClasses = labels.getRows();
```

We then define a model using the library's symbolic definition builder class `ModelBuilder`. This
has three important methods `var` to define model input variables, `param` to define learnable
parameters, `operation` to define symbolic operations and `out` to define output variables.
Most operations of this class return the same instance, so that models are incrementally builded.
For our example, we explicitly define a logistic regression model that performs a linear transformation
without bias of inputs *x* using a learnable transformation matrix *w1* and obtaining outputs *yhat*.
More operations can be handled by the mode

```java
ModelBuilder modelBuilder = new ModelBuilder()
		.var("x")
		.param("w1", new DenseMatrix(numClasses, numFeatures).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(1./dims)))
		.operation("yhat = sigmoid(w1@x)")
		.out("yhat")
		.print();
```

### Learning
```java
BatchOptimizer optimizer = new BatchOptimizer(new Regularization(new GradientDescent(0.1), 0.001));
for(int epoch=0;epoch<150;epoch++) {
	System.out.print("Epoch "+epoch);
	Tensor errors = 
			modelBuilder.getModel().trainSampleDifference(optimizer, 
			Arrays.asList(features), 
			Arrays.asList(labels))
			.get(0)
			.subtract(labels);
	
	double acc = 0;
	for(Integer node : dataset.nodes().getIds()) {
		Matrix nodeFeatures = features.getCol(node).asColumn();
		Matrix nodeLabels = labels.getCol(node).asColumn();
		Tensor output = modelBuilder.getModel().predict(nodeFeatures).get(0);
		acc += (output.argmax()==nodeLabels.argmax()?1:0);
	}
	optimizer.updateAll();
	System.out.print("\t error "+errors.abs().sum()/dataset.nodes().size());
	System.out.println("\t accuracy "+acc/dataset.nodes().size());
}
```

# :fire: Features
* Cross-platform support (written in native java)
* Speed ups with direct memory access

# :link: Material

* [Javadoc](https://maniospas.github.io/JGNN/)

* [Tutorial: Primitives](tutorials/Primitives.md) A collection of vector and matrix primitives support low-level arithmetic operations, in-place version of the same operations and both sparse and dense memory structures. The can also support applications outside the scope of the library. This tutorial covers the use of machine learning primitives.

* [Tutorial: Models and buildsers](tutorials/Models.md) Learning tasks can be automated by defining symbolic pipelines. This tutorial covers tools that can be used to create trainable machine learning models, including builders that can build those tools from  simple String expressions.

* [Tutorial: Datasets] A collection of publically available datasets that can be automatically downloaded and imported from respective external sources to facilitate benchmarking tasks. 

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker]() or by participating in [discussions]().
Please check out the [contribution guidelines](CONTRIBUTING.md) to bring modifications to the code base.
 
# :notebook: Citation