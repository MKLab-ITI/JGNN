# Training

Here we describe how to train a JGNN model created with the previous section's builders. Broadly, we need to load some reference data and employ an optimization scheme to adjust trainable parameter values based on the differences between desired and current outputs. To this end, we start by describing generic patterns for creating graph and node feature data, and then move to describing the helper class used to define and automate a training strategy.

## Create data

JGNN contains dataset classes that automatically download and load datasets for out-of-the-box experimentation. These datasets can be found in the [adhoc.datasets](https://mklab-iti.github.io/JGNN/javadoc/mklab/JGNN/adhoc/datasets/package-summary.html) Javadoc, and we already covered their usage patterns. In practice, though, you will want to use your own data. In the simplest case, both the number of nodes or data samples and the number of feature dimensions are known beforehand. If so, create dense feature matrices with the following code. This uses the minimum memory necessary to construct the feature matrix. If features are dense (do not have a lot of zeros), consider using the `DenseMatrix` class instead of initializing a sparse matrix, like below.

```java
Matrix features = new SparseMatrix(numNodes, numFeatures);
for(long nodeId=0; nodeId<numNodes; nodeId++)
    for(long featureId=0; featureId<numFeatures; featureId++)
        features.put(nodeId, featureId, 1);
```

Sometimes, it is easier to read node or sample features line-by-line, for instance, when reading a .csv file. In this case, store each line as a separate tensor. Convert a list of tensors representing row vectors into a feature matrix like in the example below.

```java
ArrayList<Tensor> rows = new ArrayList<Tensor>();
try(BufferedReader reader = new BufferedReader(new FileReader(file))){
    String line = reader.readLine();
    while (line != null) {
        String[] cols = line.split(",");
        Tensor features = new SparseTensor(cols.length);
        for(int col=0;col<cols.length;col++)
            features.put(col, Double.parseDouble(cols[col]));
        rows.add(features);
        line = reader.readLine();
    }
}
Matrix features = new WrapRows(rows).toSparse();
```

Creating adjacency matrices is similar to creating preallocated feature matrices. When in doubt, use the sparse format for adjacency matrices, as the allocated memory of dense counterparts scales quadratically to the number of nodes. Note that many GNNs consider bidirectional (i.e., non-directed) edges, in which case both directions should be added to the adjacency. Use the following snippet as a template. Recall that JGNN follows a function chain notation, so each modification returns the matrix instance. Don't forget to normalize or apply the renormalization trick (self-edges) on matrices if these are needed by your architecture, for instance by calling `adjacency.setMainDiagonal(1).setToSymmetricNormalization();` after matrix creation.

```java
Matrix adjacency = new SparseMatrix(numNodes, numNodes);
for(Entry<Long, Long> edge : edges)
    matrix
        .put(edge.getKey(), edge.getValue(), 1)
        .put(edge.getValue(), edge.getKey(), 1);
```

All tensor operations can be viewed in the [core.tensor]() and [core.matrix]() package Javadoc. The `Matrix` class extends the concept of tensors with additional operations, like transposition, matrix multiplication, and row and column access. Under the hood, matrices linearly store elements and use computations to transform the (row, col) position of their elements to respective positions. The outcome of some methods inherited from tensors may need to be typecast back into a matrix (e.g., for all in-place operations).


Operations can be split into arithmetics that combine the values of two tensors to create a new one (e.g., `Tensor add(Tensor)`) , in-place arithmetics that alter a tensor without creating a new one (e.g., `Tensor selfAdd(Tensor)`), summary statistics that output simple numeric values (e.g., `double Tensor.sum()`), and element getters and setters. In-place arithmetics follow the same naming conventions of base arithmetics but their method names begin with a "self" prefix for pairwise operations and a "setTo" prefix for unary operations. Since they do not allocate new memory, prefer them for intermediate calculation steps. For example, the following code can be used for creating and normalizing a tensor of ones without using any additional memory.

```java
Tensor normalized = new DenseTensor(10)
    .setToOnes()
    .setToNormalized();
```

Initialize a dense or sparse tensor—both of which represent one-dimensional vectors—with its number of elements. If there are many zeros expected, prefer using a sparse tensor. For example, one-hot encodings for classification problems can be generated with the following code. This creates a dense tensor with `numClasses` elements and puts at element `classId` the value 1.

```java
int classId = 1;
int numClasses = 5;
Tensor oneHotEncoding = new mklab.JGNN.tensor.DenseTensor(numClasses).set(classId, 1); // creates the tensor [0,1,0,0,0]
```

The above snippets all make use of numerical node identifiers. To manage these, JGNN provides an IdConverter class; convert hashable objects (typically strings) to identifiers by calling `IdConverter.getOrCreateId(object)`. Also use converters to one-hot encode class labels. To search only for previously registered identifiers, use `IdConverter.get(object)`. For example, construct a label matrix with the following snippet. In this, nodeLabels is a dictionary from node identifiers to node labels that is being converted to a sparse matrix.

```java
IdConverter nodeIds = new IdConverter();
IdConverter classIds = new IdConverter();
for(Entry<String, String> entry : nodeLabels) {
    nodeids.getOrCreateId(entry.getKey());
    classIds.getOrCreateId(entry.getValue());
}
Matrix labels = new SparseMatrix(nodeIds.size(), classIds.size());
for(Entry<String, String> entry : nodeLabels) 
    labels.put(nodeids.get(entry.getKey()), classIds.get(entry.getValue()), 1);
```

Reverse-search the converter to obtain the original object of predictions per `IdConverter.get(String)`. The following example accesses one row of a label matrix, performs and argmax operation to find the position of the maximum element, and reconstruct the label for the corresponding row with reverse-search.

```java
long nodeId = nodeIds.get("nodeName");
Tensor prediction = labels.accessRow(nodeId);
long predictedClassId = prediction.argmax();
System.out.println(classIds.get(predictedClassId));
```

## Model trainer

Node classification models can be backpropagated by considering a list of node indexes and desired predictions for those nodes. We first show an automation of the training process that controls it in a predictable manner.

```java
Slice nodes = dataset.samples().getSlice().shuffle(100);  // or nodes = new Slice(0, numNodes).shuffle(100);
Model model = modelBuilder()
    .getModel()
    .init(new XavierNormal())
    .train(trainer,
        nodes.samplesAsFeatures(), 
        dataset.labels(), 
        nodes.range(0, trainSplit), 
        nodes.range(trainSplit, validationSplit));
```