# :zap: Graph neural networks for graph classification
Most neural network architectures are designed with the idea of learning to classify
nodes or samples. However, GNNs also provide the prospect of classifying graphs
based on their structure.

1. [Organizing data](#organizing-data)
2. [Defining the architecture](#defining-the-architecture)
3. [Training the architecture](#training-the-architecture)

## Organizing data

To define architectures for graph classification,
we can make use of the generic `LayeredBuilder` architecture to define
a simple neural architecture for classifying graphs. The main difference compared
to traditional neural training is that training inputs do not all exhibit the 
same size (e.g. some graphs may have more nodes than others) and therefore they
can not be organized into tensors of common dimensions.

Instead, let us presume that graphs to train from are to be stored in the following lists:

```java	
ArrayList<Matrix> adjacencyMatrices = new ArrayList<Matrix>();
ArrayList<Matrix> nodeFeatures = new ArrayList<Matrix>();
ArrayList<Tensor> graphLabels = new ArrayList<Tensor>();
```

## Defining the architecture

The `LayeredBuilder` already introduce the input variable *h0* for sample features.
We can use to it to pass node features to the architectures, so we only need to add 
a second input storing the (sparse) adjacency matrix per `.var("A")`. We can proceed
to define a GNN architecture, for instance as explained in 

This time, though, we do not aim to classify nodes but the whole graph. For this reason,
we need to pool top layer node representations, for instance by averaging them
across all nodes per `.layer("h{l+1}=softmax(mean(h{l}, row))")`. Note that we apply
the softmax per normal. Finally, we need to set up the top layer as the built model's
output per `.out("h{l}")`. 

An example architecture following these principles is the following:

```java
ModelBuilder builder = new LayeredBuilder()
	.var("A")  
	.config("features", nodeLabelIds.size())
	.config("classes", graphLabelIds.size())
	.config("hidden", 16)
	.layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden)))") 
	.layer("h{l+1}=relu(A@(h{l}@matrix(hidden, classes)))") 
	.layer("h{l+1}=softmax(mean(h{l}, row))")
	.out("h{l}"); 
```

## Training the architecture

For the time being, training architectures like the above on the prepared data needs to
manually call the backpropagation for each epoch and each graph in the training
batch. To do this, we first retrieve the model and initialize its parameters:

```java
Model model = builder.getModel().init(new XavierNormal());
```

Next, we need to define a loss function and set up a batch optimization
strategy wrapping any base optimizer and accumulating parameter updates until
`BatchOptimizer.updateAll()` is called later on:

```java
Loss loss = new CategoricalCrossEntropy();
BatchOptimizer optimizer = new BatchOptimizer(new Adam(0.01));
```

Finally, training can be conducted by iterating through epochs and training samples
and appropriately calling the `Model.train` for combinations of of node features and graph
adjacency matrix inputs, and graph label outputs.
At the end of each batch (e.g. each epoch), do not forget
to call the `optimizer.updateAll()` method to apply the accumulated gradients. This
process can be realized with the following code:

```java
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

[NEXT: Data creation](Data.md)