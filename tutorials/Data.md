# :zap: Data creation

If you have been following the tutorial, we have only used automatically downloaded datasets till now.
In practice, you will want to use your own data. Thi tutorial covers typical code patterns on doing so.

1. [Creating preallocated feature matrices](#creating-preallocated-feature-matrices)
2. [Converting lists of tensors to matrices](#converting-lists-of-tensors=to-matrices)
3. [Constructing graph adjacency matrices](#constructing-graph-adjacency-matrices)
4. [Managing identifiers](#managing-identifiers)

## Creating preallocated feature matrices
If you know the number of nodes or data samples and features a-priori, you can create
dense feature matrices with the following code. This uses the bare minimum memory necessary
to construct the feature matrix. If features are dense (do not have a lot of zeroes), 
you could also consider using the `DenseMatrix` class instead of initializing a sparse matrix
- the two classes are interoperable and have the same constructor arguments
 so that the rest of the code remains the same.

```java
Matrix features = new SparseMatrix(numNodes, numFeatures);
for(long nodeId=0; nodeId<numNodes; nodeId++)
	for(long featureId=0; featureId<numFeatures; featureId++)
		features.put(nodeId, featureId, 1);
```

## Converting lists of tensors to matrices
Sometimes, it is easier to read node or sample features line-by-line, for instance as you
read a *.csv* file. In this case, you can store each line in an individual tensor. You 
can do this with a snippet like the following one, appropriately adjusted to ignore
classes or sample identifiers in your data:

```java
ArrayList<Tensor> rows = new ArrayList<Tensor>();
try(BufferedReader reader = new BufferedReader(new FileReader(file))){
	String line = reader.readLine();
	while (line != null) {
		String[] cols = line.split(",");
		Tensor features = new SparseTensor(cols.length); // or a dense tensor
		for(int col=0;col<cols.length;col++)
			features.put(col, Double.parseDouble(cols[col]));
		rows.add(features);
		line = reader.readLine();
	}
}
```

Then, the list of row tensors can be converted into a feature matrix with the expression:

```java
Matrix features = new WrapRows(rows).toSparse(); // or toDense
```

## Constructing graph adjacency matrices
Creating adjacency matrices is similar to creating preallocated features matrices. 
**Always** use the sparse format for adjacency matrices.
Note that many GNNs work by considering bidirectional (i.e. non-directed) edges,
so that if you add an edge you also need to add the oposite one.

An example snippet on how to create a symmetric adjacency matrix:

```java
Matrix adjacency = new SparseMatrix(numNodes, numNodes);
for(Entry<Long, Long> edge : edges)
	matrix.put(edge.getKey(), edge.getValue(), 1).put(edge.getValue(), edge.getKey(), 1);
```

:bulb: Don't forget to normalize or apply the renormalization trick (self-edges) on matrices 
if these are needed by your algorithm, for instance by calling `adjacency.setMainDiagonal(1).setToSymmetricNormalization();`

## Managing identifier
The above snippets all reference node identifiers. To help you with managing these, JGNN
provides an `IdConverter` class. You can convert hashable objects (e.g. Strings) to identifiers
by calling `IdConverter.getOrCreateId(object)`. The same functionality is also helpful 
for one-hot encoding of class labels. If you want to search only for previously registered identifiers, 
for example to catch logical errors, you can use `IdConverter.get(object)`.

For example, you can construct a label matrix of one-hot encodings for your training data per:

```java
// register the ids in data
IdConverter nodeIds = new IdConverter();
IdConverter classIds = new IdConverter();
for(Entry<String, String> entry : nodeLabels) {
	nodeids.getOrCreateId(entry.getKey()); // or .get(entry.getKey()) if reusing nodeIds of feature loading
	classIds.getOrCreateId(entry.getValue());
}
// create the matrix
Matrix labels = new SparseMatrix(nodeIds.size(), classIds.size());
for(Entry<String, String> entry : nodeLabels) 
	labels.put(nodeids.get(entry.getKey()), classIds.get(entry.getValue()), 1);
```

As a final remark, you can reverse-search the `IdConverter` to obtain the original object of your
predictions using the `IdConverter.get(long identifier)` to retrieve the identifier. For example:

```java
long nodeId = nodeIds.get("nodeName");
Tensor prediction = labels.accessRow(nodeId);
long predictedClassId = prediction.argmax();
System.out.println(classIds.get(predictedClassId));
```



[NEXT: Primitives](Primitives.md)