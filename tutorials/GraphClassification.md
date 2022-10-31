# :zap: Graph neural networks for graph classification
Most neural network architectures are designed with the idea of learning to classify
nodes or samples. However, GNNs also provide the prospect of classifying graphs
based on their structure.

1. [Organizing data](#organizing-data)
2. [Defining the architecture](#defining-the-architecture)
3. [Training the architecture](#training-the-architecture)
4. [Sort pooling](#sort-pooling)
5. [Parallelized training](#parallelized training)

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

## Sort pooling

Up to now, the example code performs a simple mean pooling across all graph
node features. However, this can prove insufficient for the top layers and
more sophisticated pooling mechanisms can be deployed to let GNNs differentiate
between the structural positioning of nodes to be pooled. 

One computationally light approach to pooling, which JGNN implements, is sorting
nodes based on learned features before concatenating their features in one
vector for each graph. This process if further simplified by keeping the top *reduced*
number of nodes to concatenate their features, where the order is determined by
an arbitrarily selected feature (in our implementation: the last one, with the previous
feature being used to break ties and so on). The idea is the the selected feature
determines *important* nodes whose information can be adopted by others.

To apply the following operations JGNN provides independent node sorting, gathering
node latent representations and reshaping matrices into row or column tensors with
learnable transformations to class outputs. These components are demonstrated in the 
following code snippet:

```java
long reduced = 5;  // input graphs need to have at least that many nodes,  lower values decrease accuracy
long hidden = 8;  // since this library does not use GPU parallelization, many latent dims reduce speed

ModelBuilder builder = new LayeredBuilder()        
        .var("A")  
        .config("features", 1)
        .config("classes", 2)
        .config("reduced", reduced)
        .config("hidden", hidden)
        .layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden))+vector(hidden))")  // don't forget to add bias vectors to dense transformations
        .layer("h{l+1}=relu(A@(h{l}@matrix(hidden, hidden))+vector(hidden))") 
        .concat(2) // concatenates the outputs of the last 2 layers
        .config("hiddenReduced", hidden*2*reduced)  // 2* due to concatenation
        .operation("z{l}=sort(h{l}, reduced)")  // currently, the parser fails to understand full expressions within the next step's gather, so we need to create this intermediate variable
        .layer("h{l+1}=reshape(h{l}[z{l}], 1, hiddenReduced)") //
        .layer("h{l+1}=h{l}@matrix(hiddenReduced, classes)")
        .layer("h{l+1}=softmax(h{l}, row)")
                .out("h{l}");  
```



## Parallelized training

A final aspect to keep in mind for graph classification is that you can make 
use of JGNN's parallelization capabilities to calculate gradients across 
multiple threads. Doing this for tasks like node classification holds little
meanining, as the same propagation mechanism needs to be run on the same 
graph in parallel. But this process yields substantial speedup for *graph*
classificaiton.

Parallelization can make use of the thread pooling JGNN provides to perform
gradients, wait for the conclusion of submitted tasks, and then apply all gradient
updates. This is achieved by declaring a batch optimizer to gather all the gradients.

The whole process is detailed in the following example:

```java
for(int epoch=0; epoch<500; epoch++) {
  // gradient update
  for(int graphId=0; graphId<dtrain.adjucency.size(); graphId++) {
    int graphIdentifier = graphId;
    ThreadPool.getInstance().submit(new Runnable() {
      @Override
      public void run() {
        Matrix adjacency = dtrain.adjucency.get(graphIdentifier);
        Matrix features= dtrain.features.get(graphIdentifier);
        Tensor graphLabel = dtrain.labels.get(graphIdentifier).asRow();  // Don't forget to cast to the same format as predictions.
        model.train(loss, optimizer, 
		            Arrays.asList(features, adjacency), 
		            Arrays.asList(graphLabel));
      }
    });
  }
  ThreadPool.getInstance().waitForConclusion();  // waits for all gradients to finish calculating
  optimizer.updateAll();
  
  double acc = 0.0;
  for(int graphId=0; graphId<dtest.adjucency.size(); graphId++) {
    Matrix adjacency = dtest.adjucency.get(graphId);
    Matrix features= dtest.features.get(graphId);
    Tensor graphLabel = dtest.labels.get(graphId);
    if(model.predict(Arrays.asList(features, adjacency)).get(0).argmax()==graphLabel.argmax())
       acc += 1;
    System.out.println("iter = " + epoch + "  " + acc/dtest.adjucency.size());
  }
}
```



[NEXT: Data creation](Data.md)