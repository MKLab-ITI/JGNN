# :zap: Graph neural networks

Graph neural networks (GNNs) extend the concept of base [neural networks](tutorials/NN.md).
You can already write any GNN with the base the `LayerBuilder` class for designing neural models, 
but JGNN provides some common design choices that simplify the process for node classification.

1. [Initializing a GNN builder](#initializing-a-gnn-builder)
2. [GNN concepts](#gnn-concepts)
3. [Adding a classification layer](#adding-a-classification-layer)
3. [Example architecture](#example-architecture)
5. [GNN training](#gnn-training)


## Initializing a GNN builder
The `GCNBuilder` class for building GNN architectures extends the generic 
`LayerBuilder` with common graph neural network operations. 
The only difference is that now we initialize it with a
square matrix A, which is typically a normalization of the adjacency matrix, and a feature matrix h0 
(this is different than the symbol h{0}). 
Given that you will most likely use normal neural layers, you only need 
to remember that in symbolic parsing A will correspond to the adjacency matrix
and that layer representations should be annotated with h{l}. We may make a more 
customizeable version of the builder in the future, but these symbols will always remain
the default. Preferrably, each row of the feature matrix should correspond to the features
of one node/sample. The normalized adjacency matrix can -and usually should-
be sparse to save on memory.

Most GNNs perform the renormalization trick by adding a self-loop
before applying symmetric normalization on the adjacency matrix.
Assuming no existing self-loops, the following snippet shows how to apply those
transformations on adjacency matrices, such as ones obtained from `Dadaset.graph()`. 
The snippet use in-place arithmetics to directly alter raw matrix data:

```java
adjacency.setMainDiagonal(1).setToSymmetricNormalization();
```

Finally, you can instantiate the builder by providing the adjacency and feature
matrices per:

```java
GCNBuilder modelBuilder = new GCNBuilder(adjacency, features);
```

Sending specific tensors to the builder's consructor
does not restrict you from editing or replacing them later, 
even after architectures have been trained.
For example, you can add node edges later by editing an element of the 
adjacency matrix per:

```java
Matrix adjacency = ((Constant)modelBuilder.get("A")).get(); // retrieves constant's value from the architecture
adjacency.put(from, to, value);
```


## GNN concepts

The base operation of GNNs is to propagate node representations to neighbors via graph edges,
where they are aggregated - typically summed per normalized adjacency matrix edge weights.
This can be achieved with a simple matrix multiplication on the previous layer's
node features per `.layer("h{l+1}=A @ h{l}")`. In practice, you will often want to 
add more operations on the propagation, such as passing it through a dense layer. 
For example, the original GCN architecture defines layers of the form:

```java
.layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden, reg))+vector(hidden))")
.layer("h{l+1}=A@(h{l}@matrix(hidden, classes, reg))+vector(classes)")			
```

Most architectures nowadays also perform edge dropout, which is as simple as applying dropout
on the adjacency matrix values on each layer per:

```java
.layer("h{l+1}=dropout(A,0.5) @ h{l}")
```

Recent areas of heterogenous graph research also explicitly use the graph laplacian,
which you can insert into the architecture as a normal constant per `.constant("L", adjacency.negative().cast(Matrix.class).setMainDiagonal(1))`. Even more complex concepts 
can be modelled with edge attention that gathers and
perform the dot product of edge nodes to provide new edge weights, exponentiating
non-zero weights with *nexp* and applying row-wise L1 transformation. This yields
an adjacency matrix weighting unique to the layer per `.operation("A{l}" = L1(nexp(att(A, h{l})))")`.
Nonetheless, it is recommended that you stay away from these kinds complex architectures
when learning from large graphs, as JGNN is designed to be lightweight and not fast.
Consider using GPU GNNs if 1-2% accuracy gains matter enough to make your application
several folds slower.


## Adding a classification layer
This far, we touched on propagation mechanisms of GNNs, which consider the features of all nodes.
However, when moving to a node classification setting,
training data labels are typically available only for certain nodes.
We thus need a mechanism that can retrieve the predictions of the top neural layer for certain nodes
and pass them through a softmax activation.
This can already be achieved with normal neural model definitions using the gather bracket operation
after declaring a variable of which nodes to retrieve:

```java
.var("nodes")
.layer("h{l} = softmax(h{l})")
.operation("ouput = h{l}[nodes]")
```

Recall that h{l} always points to the top layer when writting a new layer.


This way, the built model takes as inputs a set of nodes, perform the forward pass of the
architecture and then selects the provided nodes to use as outputs (and backpropagate from).
**All** nodes are needed for training because they are made aware of each other via the
graph's structure.

To simplify how node classification architectures are defined,
the above symbolic snippet is automatically generated and applied by calling the 
`.classify()` method of the `GCNBuilder` instead.

## Example architecture

As an example of how to define a full GNN with symbolic parsing, let us define
the well-known APPNP architecture. This comprises two normal dense layers and then
propagates their predictions through the graph structure with a fixed-depth approximation
of the personalized PageRank algorithm. To define the architecture,
let us consider a `Dataset dataset` loaded by the library, for which we normalize the 
adjacency matrix and send everything to the GNN builder class. We let the outcome of
the first two dense layers to be remembered as `h{0}` (this is *not* `h0`), define 
a diffusion rate constant `a` and then perform 10 times the 
personalized PageRank diffusion scheme on a graph with edge dropout 0.5. This is all achieved
with the same `layer` and `layerRepeat` methods as neural builders.

```java
dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();
long numClasses = dataset.labels().getCols();

ModelBuilder modelBuilder = new GCNBuilder(dataset.graph(), dataset.features())
				.config("reg", 0.005)
				.config("hidden", 16)
				.config("classes", numClasses)
				.layer("h{l+1}=relu(h{l}@matrix(features, hidden, reg)+vector(hidden))")
				.layer("h{l+1}=h{l}@matrix(hidden, classes)+vector(classes)")
				.rememberAs("0")
				.constant("a", 0.9)
				.layerRepeat("h{l+1} = a*(dropout(A, 0.5)@h{l})+(1-a)*h{0}", 10)
				.classify();
```


## GNN training

GNN classification models can be backpropagated by considering a list of node indeces and desired
predictions for those nodes. However, you can also use the interfaces discussed in the
[learning](tutorials/Learning.md) tutorial to automate the training process and control it
in a fixed manner. 

Recall that training needs to call the model's method 
`.train(optimizer, features, labels, train, valid)`.
The important question is what to consider as training inputs and outputs, given that node features
and the graph are passed to the `GCNBuilder` constructor.

The answer is that the (ordered) list of all node identifiers *0,1,2,...* constitutes the training inputs
and the corresponding labels constitute the outputs. You can create a slice of identifiers 
and you can use JGNN to design the training process per:

```java
Slice nodes = dataset.samples().getSlice().shuffle(100);  // or nodes = new Slice(0, numNodes).shuffle(100);
Model model = modelBuilder()
	.getModel()
	.init(...)
	.train(trainer,
			nodes.samplesAsFeatures(), 
			dataset.labels(), 
			nodes.range(0, trainSplit), 
			nodes.range(trainSplit, validationSplit));

```

In the above snipper, the label matrix can have zeroes for the nodes not used for training.
If only the first nodes have known labels, the label matrix may also have less rows.



[NEXT: Primitives](Primitives.md)