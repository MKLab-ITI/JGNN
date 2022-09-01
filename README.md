# JGNN
A native Java library for Graph Neural Networks.

# :dart: About
Graph Neural Networks (GNNs) are getting more and more popular;
they can reason from relational information and perform inference from small datasets.
JGNN provides GNNs to native Java applications and supports cross-platform machine
learning, such as on Android, without specific hardware or firmware.

# :zap: Quickstart
To install the latest working version, you can include it as Maven or Gradle dependency by following the instructions of its JitPack distribution:
[![](https://jitpack.io/v/maniospas/jgnn.svg)](https://jitpack.io/#maniospas/jgnn)

```java
Dataset dataset = new Datasets.CiteSeer();
Matrix labels = dataset.nodes().oneHot(dataset.getLabels());
Matrix features = dataset.nodes().oneHotFromBinary(dataset.getFeatures());

Matrix adjacency = new SparseMatrix(dataset.nodes().size(), dataset.nodes().size());
for(Entry<Long, Long> interaction : dataset.getInteractions()) 
	adjacency
		.put(interaction.getKey(), interaction.getValue(), 1)
		.put(interaction.getValue(), interaction.getKey(), 1);
adjacency.setMainDiagonal(1).setToSymmetricNormalization();

long numClasses = labels.getCols();
ModelBuilder modelBuilder = new GCNBuilder(adjacency, features)
		.config("reg", 0.005)
		.config("classes", numClasses)
		.config("hidden", hidden)
		.layer("h{l+1}=relu(h{l}@matrix(features, hidden, reg)+vector(hidden))")
		.layer("h{l+1}=h{l}@matrix(hidden, classes)+vector(classes)")
		.rememberAs("0")
		.constant("a", 0.9)
		.layerRepeat("h{l+1} = a*(dropout(A, 0.5)@h{l})+(1-a)*h{0}", 10)
		.classify();				;

ModelTraining trainer = new ModelTraining()
		.setOptimizer(new Adam(0.01))
		.setEpochs(300)
		.setPatience(100)
		.setLoss(new CategoricalCrossEntropy())
		.setValidationLoss(new Accuracy());

Slice nodes = dataset.nodes().getIds().shuffle(100);
Model model = modelBuilder.getModel()
		.init(new XavierNormal())
		.train(trainer,
				Tensor.fromRange(0, nodes.size()).asColumn(), 
				labels, nodes.range(0, 0.2), nodes.range(0.2, 0.4));


Matrix output = model.predict(Tensor.fromRange(0, nodes.size()).asColumn()).get(0).cast(Matrix.class);
double acc = 0;
for(Long node : nodes.range(0.4, 1)) {
	Matrix nodeLabels = labels.accessRow(node).asRow();
	Tensor nodeOutput = output.accessRow(node).asRow();
	acc += nodeOutput.argmax()==nodeLabels.argmax()?1:0;
}
System.out.println("Acc\t "+acc/nodes.range(0.4, 1).size());
```

# :rocket: Features
* Cross-platform (written in native java)
* Lightweight
* Minimal memory footprints
* Parallelized batching

# :link: Tutorials
* [Tutorials](tutorials/Tutorials.md)
* [Javadoc](https://maniospas.github.io/JGNN/)

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/JGNN/issues).
Please check out the [contribution guidelines](CONTRIBUTING.md) 
to bring modifications to the code base.
 
# :notebook: Citation
TBD
