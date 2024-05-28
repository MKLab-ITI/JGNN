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
fn classify(nodes, h, epochs: !3000, patience: !100, lr: !0.01) {
	return softmax(h[nodes], dim: "row");
}
```

The classify function takes two inputs: nodes are the input nodes for classification; h is the feature matrix. The function returns a softmax output for the specified nodes. It also considers several configuration values, whose defaults are indicated by a colon (:) in the function signatures. The same notation is used to set/overwrite them when calling functions, as we do for softmax to apply it row-wise. Think of them as keyword arguments. These defaults for the classify function are: epochs, which defaults to 3000 and represents the number of training epochs; patience, which defaults to 100 and denotes the early stopping patience; and lr the learning rate that defaults to 0.01. 

Exclamation marks (!) before numbers broadcast them to all subsequent function calls as new defaults for the same configurations. Broadcasted configurations are retrievable from JGNN's Neuralang model builder, which is useful for Java integration later. Configuration values have the priority: 
1. function call arguments
2. broacasted configurations (last value, includes configurations set by Java)
3. function signature defaults

```rust
fn gcnlayer(A, h, hidden: 64, reg: 0.005) {
	return relu(A@h@matrix(?, hidden, reg) + vector(hidden));
}
```

The gcnlayer function accepts the following parameters: A is the adjacency matrix; h is the input feature matrix; hidden is a configuration that defaults to 64 and specifies the number of hidden units; and reg is the regularization term that defaults to 0.005. The ? in matrix definitions lets the autosize feature of Java integration determine the dimension name based on a test run, which uses an empty tensor to avoid computations. The function returns the activated output of the GCN layer using ReLU.

```rust
fn gcn(A, h, classes: extern) {
	h = gcnlayer(A, h);
	h = gcnlayer(A, h, hidden: classes);
	return h;
}
```

The gcn function declares the popular Graph Convoluational Network (GCN) architecture. It takes these parameters: A is the adjacency matrix; h is the input feature matrix; and classes is the number of output classes. The function first applies a gcnlayer to A and h, and then applies another layer of the same type with the hidden units configuration set to the value of classes, to make the output match the number of classes. The number of classes is a config that should always be externally declared (there is no default), as indicated by its value being the extend keyword.


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

In the above example, a dataset (Cora) is loaded, and its graph is prepared by adding self-loops (the renormalization trick) and performing symmetric normalization. A Neuralang instance is ten created; this is a ModelBuilder that can parse scripts as either file Paths or pure text. Constants like the adjacency matrix A and feature matrix h are set, along with variables (nodes) and configurations (classes, hidden). The model and its output is defined with a Neuralang statement. Finally, dimension names and sizes for ? found model declaration are filled by calling autosize. In the example we use empty tensors to avoid unecessary computations while determining the dimensions.

A ModelTraining instance is finally configured using parameters from the ModelBuilder, utilizing the configurations found in the classification method. Don't forget to broadcast configuration values that you need to access from Java code later.
