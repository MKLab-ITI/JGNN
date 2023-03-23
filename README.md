# JGNN
A native Java library for Graph Neural Networks.

# :cyclone: Changes from earlier versions
* Renamed `GCNBuilder` to `FastBuilder`
* Added attention and message passing components.
* Added sort pooling and graph classification.

# :dart: About
Graph Neural Networks (GNNs) are getting more and more popular;
they can reason from relational information and perform inference from small datasets.
JGNN provides GNNs to native Java applications and supports cross-platform machine
learning, such as on Android, without specific hardware or firmware.

* [Tutorials](tutorials/README.md)
* [Javadoc](https://mklab-iti.github.io/JGNN/)

# :rocket: Features
* Cross-platform (written in native java)
* Lightweight
* Minimal memory footprints
* Symbolic model definition
* Parallelized batching

# :zap: Quickstart
To install the latest working version, you can include it as Maven or Gradle dependency by following the instructions of its JitPack distribution:
[![](https://jitpack.io/v/MKLab-ITI/JGNN.svg)](https://jitpack.io/#MKLab-ITI/JGNN)

As a good out-of-the-box graph neural network for node classification, you can try the 
following architecture. To create your own graph adjacency matrices,
node features, and labels look at the [data creation](tutorials/Data.md) tutorial.

```java
Dataset dataset = new Cora();
Matrix adjacency = dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();
Matrix nodeFeatures = dataset.features();
Matrix nodeLabels = dataset.labels();
Slice nodes = new Slice(0, nodeLabels.getRows()).shuffle(100);
long numClasses = nodeLabels.getCols();

ModelBuilder modelBuilder = new FastBuilder(adjacency, nodeFeatures)
		.config("reg", 0.005)
		.config("hidden", 16)
		.config("classes", numClasses)
		.layer("h{l+1}=relu(h{l}@matrix(features, hidden, reg)+vector(hidden))")
		.layer("h{l+1}=h{l}@matrix(hidden, classes)+vector(classes)")
		.rememberAs("0")
		.constant("a", 0.9)
		.layerRepeat("h{l+1} = a*(dropout(A, 0.5)@h{l})+(1-a)*h{0}", 10)
		.classify();

ModelTraining trainer = new ModelTraining()
		.setOptimizer(new Adam(0.01))
		.setEpochs(300)
		.setPatience(100)
		.setLoss(new CategoricalCrossEntropy())
		.setValidationLoss(new CategoricalCrossEntropy());

Model model = modelBuilder.getModel()
		.init(new XavierNormal())
		.train(trainer,
				nodes.samplesAsFeatures(), 
				nodeLabels, 
				nodes.range(0, 0.6), 
				nodes.range(0.6, 0.8));

Matrix output = model.predict(nodes.samplesAsFeatures()).get(0).cast(Matrix.class);
double acc = 0;
for(Long node : nodes.range(0.8, 1)) {
	Matrix nodeLabels = dataset.labels().accessRow(node).asRow();
	Tensor nodeOutput = output.accessRow(node).asRow();
	acc += nodeOutput.argmax()==nodeLabels.argmax()?1:0;
}
System.out.println("Acc\t "+acc/nodes.range(0.8, 1).size());
```

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/JGNN/issues).
Please check out the [contribution guidelines](CONTRIBUTING.md) 
when bringing modifications to the code base.
 
# :notebook: Citation
TBD
