# :zap: Graph neural networks

Graph neural networks (GNNs) extend the concept of base [Neural Networks](tutorials/NN.md).
You can write any GNN with the base neural model builder (`LayerBuilder`), but JGNN provides
some premade design choices that can simplify the whole process.

1. [Initializing a GNN builder](#initializing-a-gnn-builder)
2. [GNN concepts](#gnn-concepts)
3. [Adding a classification layer](#adding-a-classification-layer)
4. [GNN training](#gnn-training)


## Initializing a GNN builder
The class for buildign GNN architectures (`GCNBuilder`) extends the generic 
`LayerBuilder` for neural networks but is instead initialized with a
normalized square adjacency matrix A and a feature matrix h0 
(this is not to be confused with any potential h{0}). 
Given that you will most likely use layer definitions, you only need 
to remember that in symbolic parsing A corresponds to the adjacency matrix
and that layer representations should be annotated with h. We may make a more 
customizeable version of the builder in the future, but this will always be
the default. Preferrably, each row of the feature matrix should correspond to the features
of one node/sample. The adjacency matrix can -and usually should-
be sparse to save on memory.

Most GNNs perform the renormalization trick (by adding 1 to the diagonal) and
perform symmetric normalization of the adjacency matrix too.
Assuming no existing self-loops, the following snippet shows how to apply those
concepts on existing adjacency matrices. Note that in-place arithmetics are used,
with means that the raw matrix data are directly altered:

```java
adjacency.setMainDiagonal(1).setToSymmetricNormalization();
```

Finally, you can instantiate the builder by providing the adjacency and feature
matrives per:

```java
GCNBuilder modelBuilder = new GCNBuilder(adjacency, features);
```

Sending specific tensors to the builder's consructor
does not restrict you from editing or replacing them later, 
even after architectures have been trained.
For example, you can add node edges later by editing the adjacency matrix
per:

```java
((Constant)modelBuilder.get("A")).put(from, to, value);
```

## GNN concepts
The base operation of GNNs is to propagate node information to node neighbors.
Normally this can be achieved with a simple matrix multiplication on the previous layer
node features per:

```java
.layer("h{l+1}=A @ h{l}")
```

Many architectures also perform edge dropout, which is as simply as applying dropout
on the adjacency matrix on each layer, for example per:

```java
.layer("h{l+1}=dropout(A,0.5) @ h{l}")
```

Recent areas of heterogenous graph research also explicitly use the graph laplacian,
which you can insert into the architecture as a normal constant per:

```java
.constant("L", adjacency.negative().cast(Matrix.class).setMainDiagonal(1))
```

More complex concepts can also be modelled with edge attention that gathers and
perform the dot product of edge nodes to provide new edge weights, exponentiating
non-zero weights with *nexp* and applying row-wise L1 transformation. This yields
an adjacency matrix weighting unique to the layer per:

```java
.operation("A{l}" = L1(nexp(att(A, h{l})))")
```

:warning: It is recommended that you stay away from these kinds complex architectures
when learning from large graphs, as JGNN is designed to be lightweight and not fast.
Consider using GPU GNNs if 1-2% accuracy gains matter enough to make your application
several folds slower.


## Adding a classification layer
This far, we touched on propagation mechanisms of GNNs considering that all nodes 
exhibit features. However, training data are typically available only for certain nodes.
We thus need a mechanism that can retrieve the top neural layer predictions for certain nodes.
If we only had a neural network this would have been achieved with the gather operation on the
top layer's softmax activation on a set of node indexes per:

```java
.var("nodes")
.layer("h{l} = softmax(h{l})")
.operation("ouput = h{l}[nodes]")
```

This way, the built model takes as inputs a set of nodes, perform the forward pass of the
architecture and then selects the provided nodes to use as outputs (and backpropagate from).
**All** nodes are needed for training because they are made aware of each other via the
graph's structure.

The above classivation pattern is packed into an equivalent `.classify()` method that 
creates a classification layer.


## GNN training


[NEXT: Primitives](Primitives.md)