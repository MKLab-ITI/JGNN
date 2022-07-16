# :zap: Introduction to JGNN

As an example, we will create a custom model to make predictions on the Lymphography dataset.
This tutorial covers the following topics:

1. [Dataset loading](#dataset-loading)
2. [Model definition](#model-definition)
3. [Training](#training)
4. [Testing](#testing)

# Dataset loading
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

:bulb: To maintain the same naming convention between traditional and graph neural networks, data samples
are refferred to as *nodes* .

# Model definition
We then store the number of features and class labels that will help us define our model. It
is worth noting that matrix dimensions and elements are `long` numbers. Dense matrices can only
store up to integers, but this conversion allows us to handle large sparce matrices through the
same interfaces.

```java
long numFeatures = features.getRows();
long numClasses = labels.getRows();
```

We then define a model using the library's symbolic definition builder class `ModelBuilder`. This
has three important methods: `var` to define model input variables, `param` to define learnable
parameters, `operation` to define symbolic operations and `out` to define output variables.
Most operations of this class return the builder's instance, so that models are incrementally constructed.
For our example, we explicitly define a logistic regression model that performs a linear transformation
without bias of inputs *x* using a learnable transformation matrix *w1* and obtaining outputs *yhat*.
More operations can be handled by the library and can be found in the (introduction to models and builders)(#Models.md).

```java
ModelBuilder modelBuilder = new ModelBuilder()
		.var("x")
		.param("w1", new DenseMatrix(numClasses, numFeatures).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(1./dims)))
		.operation("yhat = sigmoid(w1@x)")
		.out("yhat")
		.print();
```

# Training
To train the model, we first set up a training-test split of node identifiers. We also use the `WrapCols`
subclass and the `Matrix.accessColumns` methods to access the features and labels of specifically the
training nodes without needing to re-allocate anything, i.e. the data split is lightweight.

```java
ArrayList<Long> nodeIds = dataset.nodes().getIds();
Collections.shuffle(nodeIds);
List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
Matrix trainFeatures = new WrapCols(features.accessColumns(trainIds));
Matrix trainLabels = new WrapCols(labels.accessColumns(trainIds));
```

We then create a new optimizer that performs gradient descent with learning rate 0.1 
and L2 regularization of parameters with weight 0.001. We also set up a batch optimizer, which accumulates
gradients and only updates them when its `BatchOptimizer.updateAll()` method is called.

```java
Optimizer optimizer = new Regularization(new GradientDescent(0.1), 0.001);
BatchOptimizer batchOptimizer = new BatchOptimizer(batchOptimizer);
```

:bulb: **Always** use batch optimizers for complicated models, as more than one gradient paths may arrive
at trainable variables and they need to perform simultaneous updates.

:bulb: Instead of adding a universal regularizer, you can instead set a regularization constant
to model variables by inserting it just after their name in the `param` method. This forces
L2 regularization to be applied during all gradient calculations.


We finally perform training over a total of 150 epochs. For each of these, we obtain the model held by the model builder
and call its `Model.trainSampleDifference` method to train it with a provided optimizer, list of of inputs variable
values and list of output variable values (these lists comprise only one matrix each). This method returns the original
predictions as a list. Batching and parallelized batch training are covered in [introduction to models and builders](Models.md).

:bulb: The order of input and output list elements, if more than one need to be provided or retrieved, corresponds to
their definition order in the model builder.

```java
Model model = modelBuilder.getModel();
for(int epoch=0;epoch<150;epoch++) {
	Tensor yhat = model.trainSampleDifference(batchOptimizer, Arrays.asList(features), Arrays.asList(labels)).get(0);
	Tensor errors = yhat.subtract(labels);
	batchOptimizer.updateAll();
	print("Epoch "+epoch+" error "+errors.abs().sum()/trainIds.size())
}
```

# Testing
We finally report training accuracy on the test set. We demonstrate how single-node (single-sample) predictions can be
made and measure the accuracy of those. To do this, we use `Matrix.accessCol` to obtain specific matrix columns from node
features as tensors and `Tensor.asColumn` to convert the obtained tensors into a column representation. Column representations
are matrices and hence can pass through the model's defined matrix multiplication. We finally use `argmax` to convert one-hot 
prediction encodings to label ids. Overall, this sample code achieves 82.8% accuracy (results may vary a little between experiment
runs due to random splitting).

```java
double acc = 0;
for(Integer node : testIds) {
	Matrix nodeFeatures = features.accessCol(node).asColumn();
	Matrix nodeLabels = labels.accessCol(node).asColumn();
	Tensor output = modelBuilder.getModel().predict(nodeFeatures).get(0);
	acc += (output.argmax()==nodeLabels.argmax()?1:0);
}
System.out.println("Accuracy "+acc/testIds.size());
```
