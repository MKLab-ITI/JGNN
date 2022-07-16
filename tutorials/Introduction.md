# :zap: Introduction to JGNN

As an example, we will create a custom model to make predictions on the Lymphography dataset.
This tutorial covers the following topics:

1. [Dataset loading](#dataset-loading)
2. [Model definition](#model-definition)
3. [Training](#training)
4. [Testing](#testing)

Code snippets are organized into an [Introduction.java](JGNN/src/examples/Introduction.java) file.

# Dataset loading
First, we load the dataset (it is automatically downloaded) and use the `IdConverter` class
to transform its labels and features into matrices. To save memory space, the library implicitly
makes these matrices sparse, although this is not explicitly defined. These matrices could also
be programmatically generated in either sparse or dense form (according to memory vs speed
considerations) using primitive operations.

Loaded features and labels are organized into `Matrix` instances. We also set dimension names
to force the library to check for logical integrity when performing operations (`null` names
can match anything), for example during matrix addition or multiplication.

```java
Dataset dataset = new Datasets.Lymphography();
IdConverter nodeIdConverter = dataset.nodes();
Matrix labels = nodeIdConverter.oneHot(dataset.getLabels()).setDimensionName("samples", "classes");
Matrix features = nodeIdConverter.oneHot(dataset.getFeatures()).setDimensionName("samples", "features");
```

:bulb: To maintain the same naming convention between traditional and graph neural networks, 
data samples are refferred to as *nodes* .

# Model definition
We then store the number of features and class labels that will help us define our model. It
is worth noting that matrix dimensions and elements are `long` numbers. Dense matrices can only
store up to integers, but this conversion allows us to handle large sparce matrices through the
same interfaces.

```java
long numFeatures = features.getCols();
long numClasses = labels.getCols();
```

We define a model using the library's symbolic builder class. This
has four important methods: a) `var` to define model input variables, b) `config` to define
hyperparameters used for matrix and vector construction,
c) `operation` to define symbolic operations and learnable parameters,
and d) `out` to define output variables. Later on, we can just retrieve the defined model at anytime
using the builders `getModel()` method.
Most operations return the builder's instance, so that models can be incrementally constructed.
For our example, we explicitly define a two-layer perceptron.
Notably, learnable matrices and vectors can be defined symoblically (it is possible to 
manual define them too, but this way is faster). The number of
hidden dimensions (64 right now) could also have been set as a hyperparameter. `@` corresponds
to matrix multiplication.

```java
ModelBuilder modelBuilder = new ModelBuilder()
	.config("features", numFeatures)
	.config("classes", numClasses)
	.config("regularize", 1.E-5)
	.var("x")
	.operation("h = relu(x@matrix(features, 64, regularize)+vector(64))")
	.operation("yhat = sigmoid(h@matrix(64, classes)+vector(classes))")
	.out("yhat");
```

To help check the architecture, you can extract its execution graph in *.dot* format
by writting:

```java
System.out.println(modelBuilder.getExecutionGraphDot());
```

For example, copying-and-pasting the graph description to [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline/) creates the following visualization
of the execution graph:

![Example execution graph](graphviz.png)

# Training
To train the model, we first set up a 50-25-25 training-validation-test split of data.

```java
ArrayList<Long> nodeIds = dataset.nodes().getIds();
Collections.shuffle(nodeIds);
List<Long> trainIds = nodeIds.subList(nodeIds.size()/2, nodeIds.size());
List<Long> validationIds = nodeIds.subList(nodeIds.size()/4, nodeIds.size()/2);
List<Long> testIds = nodeIds.subList(0, nodeIds.size()/2);
```

Before training the model, we initialize its parameters using the Xavier normalizer. 
The library's implementation
automatically determines non-linearity weights. For example, you don't need to determine 
different types of normalization for each neural network layer. Initialization can be
achieved per:

```java
Model model = new XavierNormal().apply(modelBuilder.getModel());
```

We finally create an Adam optimizer with learning rate 0.01 and
use this to define an experiment setting that runs parallelized
stochastic gradient descent (remember that this is still native Java!)
on a cross-entropy loss.
We set training to use a patience strategy for early stopping if
validation loss has not decreased for 100 epochs. Defining this 
training strategy can done per:


```java
Optimizer optimizer = new Adam(0.1);

ModelTraining trainer = new ModelTraining()
	.setOptimizer(optimizer)
	.setEpochs(3000)
	.setPatience(100)
	.setNumBatches(10)
	.setParallelizedStochasticGradientDescent(true)
	.setLoss(ModelTraining.Loss.CrossEntropy);
```

Finally, the model can be easily trained per:

```java
model = trainer.train(model, features, labels, trainIds, validationIds);
```

Real-world settings can further separate rows of testIds first, but we don't do this
in this example for the sake of simplicity.


# Testing
We finally report training accuracy on the test set. We demonstrate how single-sample predictions can be
made and measure the accuracy of those. To do this, we use `Matrix.accessRow` to obtain specific matrix rows from node features as tensors and `Tensor.asRow` to convert the obtained tensors into a row representation. Row representations
are matrices and hence can pass through the model's matrix multiplication. We finally use `argmax` to convert one-hot prediction encodings to label ids.

```java
double acc = 0;
for(Long node : testIds) {
	Matrix nodeFeatures = features.accessRow(node).asRow();
	Matrix nodeLabels = labels.accessRow(node).asRow();
	Tensor output = model.predict(nodeFeatures).get(0);
	acc += (output.argmax()==nodeLabels.argmax()?1:0);
}
System.out.println("Acc\t "+acc/testIds.size());
```
