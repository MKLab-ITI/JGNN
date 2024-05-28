# Neuralang

This is a scripting language for defining graph and traditional
neural network architectures. It extends JGNN's symbolic definition
with function declarations that construct the underlying execution
graph.



## Script

Neuralang scripts consist of function declarations like the ones bellow.
These scripts define neural network components and their interactions
using a syntax inspired by Mojo. Use a Rust highlighter, which covers
all keywords. Below are examples of function declarations:

```rust
fn classify(nodes, h, epochs: 3000, patience: 100, lr: 0.01) {
	return softmax(h[nodes], row);
}
```

The classify function takes several parameters: nodes, which are the input nodes for classification; h, the feature matrix; epochs, which defaults to 3000 and represents the number of training epochs; patience, which defaults to 100 and denotes the early stopping patience; and lr, the learning rate, which defaults to 0.01. The function returns the softmax output for the specified nodes. Configuration defaults are indicated by a colon (:) in the function signatures.

```rust
fn gcnlayer(A, h, hidden: 64, reg: 0.005) {
	return relu(A@h@matrix(?, hidden, reg) + vector(hidden));
}
```

The gcnlayer function accepts the following parameters: A is the adjacency matrix; h is the input feature matrix; hidden is a config that defaults to 64 and specifies the number of hidden units; and reg is the regularization term that defaults to 0.005. The ? in matrix definitions lets the autosize feature of Java integration determine the dimension name based on a test run, which uses an empty tensor to avoid computations. The function returns the activated output of the GCN layer using ReLU.

```rust
fn gcn(A, h, classes: extern) {
	h = gcnlayer(A, h);
	h = gcnlayer(A, h, hidden: classes);
	return h;
}
```

The gcn function declares the popular Graph Convoluational Network (GCN) architecture. It takes these parameters: A is the adjacency matrix; h is the input feature matrix; and classes is the number of output classes. The function first applies a gcnlayer to A and h, and then applies another layer of the same type with the hidden units config set to the value of classes, to make the output match the number of classes. The number of classes is a config that should always be externally declared (there is no default), as indicated by its value being the extend keyword.

Config values have the priority: values provided during function calls > Java configs > defaults specified in function signatures.


## Java integration

Neuralang scripts can be integrated into Java code for building and training models. Below is an example of how to do so:


```java
Dataset dataset = new Cora();
dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();

ModelBuilder modelBuilder = new Neuralang()
				.parse(Paths.get("../architectures.nn"))
				.constant("A", dataset.graph())
				.constant("h", dataset.features())
				.var("nodes")
				.config("classes", dataset.labels().getCols())
				.config("hidden", 16)
				.out("classify(nodes, gcn(A,h))")
				.autosize(new EmptyTensor(dataset.samples().getSlice().size()));

ModelTraining trainer = new ModelTraining()
				.configFrom(modelBuilder)
				.setVerbose(true)
				.setLoss(new CategoricalCrossEntropy())
				.setValidationLoss(new CategoricalCrossEntropy());
```

In this example, a dataset (Cora) is loaded, and its graph is prepared by setting the main diagonal and normalizing symmetrically. A Neuralang instance is created, which is a ModelBuilder that can parse scripts. Constants like the adjacency matrix A and feature matrix h are set, along with variables (nodes) and configurations (classes, hidden). The output of the model is defined with a Neuralang statement. Finally, dimension names and sizes for ? found model declaration are filled by calling autosize. This uses an empty tensor to avoid computations and determine the necessary dimensions.

A ModelTraining instance is then configured using parameters from the ModelBuilder, utilizing the configurations found in the classification method.
