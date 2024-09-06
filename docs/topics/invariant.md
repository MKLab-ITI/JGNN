# Invariant GNNs

Many scenarios are constrained on **equivariant** GNNs, whose outputs maintain the order of any node permutations applied to their inputs. In other words, if the order of node identifiers is modified (both in the graph adjacency matrix and in node feature matrices), the order of rows in the output will also be modified accordingly. Most operations described earlier are equivariant, so their synthesis results in equivariant models unless explicitly stated otherwise.

However, there are cases where the desired behavior is for GNNs to be **invariant**, meaning that the model's predictions should remain unchanged regardless of any permutation applied to the input nodes. This is particularly important when the task involves classifying entire graphs rather than individual nodes, as the model should produce a consistent output for the same graph structure, regardless of how its nodes are ordered.


To impose invariance in a GNN, take an existing equivariant architecture and apply an invariant operation on top. You may want to perform further transformations (e.g., using dense layers) afterward, but the core idea remains the same. JGNN provides two types of invariant operations, also known as pooling: 

1. **Reduction-based pooling** is straightforward to implement by using a dimensionality reduction mechanism, such as `min`, `max`, `sum`, or `mean`, applied *column-wise* on the output feature matrix. Since each row corresponds to a different node's features, the result of the reduction is a one-dimensional vector where each entry aggregates feature values across all nodes.

2. **Sort-based pooling** offers an alternative that sorts nodes based on learned features before concatenating their features into a single vector for each graph. This approach can be further optimized by retaining only the top *reduced* number of nodes, determined by an arbitrarily selected feature (in our implementation, the last feature is used, with the previous features serving as tiebreakers). The main idea is that the selected feature determines the "important" nodes, whose information is aggregated in the final output. To implement this scheme, JGNN provides independent operations to sort nodes, gather node latent representations, and reshape matrices into row or column tensors with learnable transformations to class outputs. The following code snippet demonstrates this approach:

```java
long reduced = 5;  // Input graphs need to have at least this many nodes
long hidden = 8;  // Many latent dimensions reduce speed without GPU parallelization

ModelBuilder builder = new LayeredBuilder()        
    .var("A")  
    .config("features", 1)
    .config("classes", 2)
    .config("reduced", reduced)
    .config("hidden", hidden)
    .layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden))+vector(hidden))")
    .layer("h{l+1}=relu(A@(h{l}@matrix(hidden, hidden))+vector(hidden))")
    .concat(2)  // Concatenates the outputs of the last 2 layers
    .config("hiddenReduced", hidden * 2 * reduced)  // 2* due to concatenation
    .operation("z{l}=sort(h{l}, reduced)")  // z{l} are node indexes
    .layer("h{l+1}=reshape(h{l}[z{l}], 1, hiddenReduced)")
    .layer("h{l+1}=h{l}@matrix(hiddenReduced, classes)")
    .layer("h{l+1}=softmax(h{l}, dim: 'row')")
    .out("h{l}");
```


## Train graph classifiers

Most neural network architectures are designed with the idea of learning to classify nodes or samples. However, GNNs also provide the capability to classify entire graphs based on their structure. To define architectures for graph classification, we use the generic LayeredBuilder class. The main difference compared to traditional neural networks is that architecture inputs do not all exhibit the same size (e.g., some graphs may have more nodes than others) and therefore cannot be organized into tensors of common dimensions. Instead, assume that training data are stored in the following lists:

```java
ArrayList<Matrix> adjacencyMatrices = new ArrayList<Matrix>();
ArrayList<Matrix> nodeFeatures = new ArrayList<Matrix>();
ArrayList<Tensor> graphLabels = new ArrayList<Tensor>();
```

The `LayeredBuilder` class introduces the input variable `h0` for sample features. We can use it to pass node features to the architecture, so we only need to add a second input storing the (sparse) adjacency matrix. 
We can then proceed to define a GNN architecture, for instance as explained in previous tutorials. This time, though, we aim to classify entire graphs rather than individual nodes. For this reason, we need to pool top layer node representations, for instance by averaging them across all nodes. Finally, set up the top layer as the built model's output.  An example architecture following these principles follows

```java
ModelBuilder builder = new LayeredBuilder()
    .var("A")  
    .config("features", nodeLabelIds.size())
    .config("classes", graphLabelIds.size())
    .config("hidden", 16)
    .layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden)))") 
    .layer("h{l+1}=relu(A@(h{l}@matrix(hidden, classes)))") 
    .layer("h{l+1}=softmax(mean(h{l}, dim: 'row'))")
    .out("h{l}");
```

For the time being, training architectures like the above on prepared data requires manually calling the backpropagation for each epoch and each graph in the training batch. To do this, first retrieve the model and initialize its parameters. Next, define a loss function and set up a batch optimization strategy wrapping any base optimizer and accumulating parameter updates until `BatchOptimizer.updateAll()` is called later on. Finally, training can be conducted by iterating through epochs and training samples and appropriately calling the Model.train for combinations of node features and graph adjacency matrix inputs and graph label outputs. At the end of each batch (e.g., each epoch), don't forget to call the `optimizer.updateAll()` method to apply the accumulated gradients. This process can be realized with the following code.

```java
Model model = builder.getModel()
    .init(new XavierNormal());
	
Loss loss = new CategoricalCrossEntropy();
BatchOptimizer optimizer = new BatchOptimizer(new Adam(0.01));

for(int epoch=0; epoch<300; epoch++) {
    for(int graphId=0; graphId<graphLabels.size(); graphId++) {
        Matrix adjacency = adjacencyMatrices.get(graphId);
        Matrix features = nodeFeatures.get(graphId);
        Tensor label = graphLabels.get(graphId);
        model.train(loss, optimizer, 
            Arrays.asList(features, adjacency), 
            Arrays.asList(label));
    }
    optimizer.updateAll();
}
```


To speed up graph classification, use JGNN's parallelization capabilities to calculate gradients across multiple threads. Parallelization for node classification holds little meaning, as the same propagation mechanism needs to be run on the same graph in parallel. However, this process yields substantial speedup for the graph classification problem. Parallelization can use JGNN's thread pooling to perform gradients, wait for the conclusion of submitted tasks, and then apply the accumulated gradient updates. This is achieved through a batch optimizer that accumulates gradients in the following example: