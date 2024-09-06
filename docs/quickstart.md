# Quickstart

Here we demonstrate usage of JGNN for node classification. This is an inductive learning task that predicts node labels given a graph's structure, node features, and some already known labels. Classifying graphs is also supported, though it is harder to explain and set up.

GNN architectures for node classification are typically written as message-passing mechanisms; they diffuse node representations across edges, where node neighbors pick up, aggregate (e.g., average), and transform incoming representations to update theirs. Alternatives that boast higher expressive power also exist and are supported, but simple architectures may be just as good or better than complex alternatives in solving practical problems ([Krasanakis et al., 2024](https://www.mdpi.com/2076-3417/14/11/4533)). Simpler architectures also enjoy reduced resource consumption.

## Node classification GNN

Our demonstration starts by loading the `Cora` dataset from those shipped with the library for out-of-the-box experimentation. The first time an instance of this dataset is created, it downloads its raw data from a web resource and stores them in a local `downloads/` folder. The data are then loaded into a sparse graph adjacency matrix, a dense node feature matrix, and a dense node label matrix.

Sparse and dense representations are interchangeable in terms of operations, with the main difference being that sparse matrices are much more efficient when they contain lots of zeros. JGNN automatically determines the types of intermediate representations, so focus only on choosing input and desired output data formats. In the loaded matrices, each row contains the corresponding node's neighbors, features, or one-hot encoding of labels. We apply the renormalization trick and symmetric normalization on the dataset's adjacency matrix using in-place operations for minimal memory footprint. The first of the two makes GNN computations numerically stable by adding self-loops to all nodes, while renormalization is required by spectral-based GNNs, such as the model we implement next.

```java
Dataset dataset = new Cora();
dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();
```

We incrementally create a trainable model using symbolic expressions that resemble math notation. The expressions are part of a scripting language, called Neuralang, that is covered in the namesake [language tutorial](tutorial/neuralang.md). For faster onboarding, stick to the `FastBuilder`, which omits some of the language's features in favor of providing programmatic shortcuts for boilerplate code. Its constructor accepts two arguments `A` and `h0`, respectively holding the graph's adjacency matrix and node features. These arguments are set as constant symbols that parsed expressions can use. Other constants and input variables can be set afterwards, but more on this later. After instantiation, use some builder methods to declare a model's data flow. Some of these methods parse the aforementioned expressions.

- **`config`** - Configures hyperparameter values. These can be used in all subsequent function and layer declarations.
- **`function`** - Declares a Neuralang function, in this case with inputs `A` and `h`.
- **`layer`** - Declares a layer that can use built-in and Neuralang functions. In this, the symbols `{l}` and `{l+1}` specifically are replaced by a layer counter.
- **`classify`** - Adds a softmax layer tailored to classification. This also silently declares an input `nodes` that represents a list of node indices where the outputs should be computed.
- **`autosize`** - Automatically sizes matrix and vector dimensions that were originally denoted with a question mark `?`. This method requires some input example, and here we provide a list of node identifiers, which we also make dataless (have only the correct dimensions without allocating memory). This method also checks for integrity errors in the declared architecture, such as computational paths that do not lead to an output.

JGNN promotes method chains, where the builder's instance is returned by each of its methods to access the next one. Below, we use this programming pattern to implement the Graph Convolutional Network (GCN) architecture ([Kipf and Welling, 2017](https://arxiv.org/abs/1609.02907)). Details on the symbolic parts of definitions are presented later but, for the time being, we point to the `matrix` and `vector` functions. These declare inline some trainable parameters for given dimensions and regularization. Access the created model via `modelBuilder.getModel()`.

```java
long numSamples = dataset.samples().getSlice().size();
long numClasses = dataset.labels().getCols();
ModelBuilder modelBuilder = new FastBuilder(dataset.graph(), dataset.features())
    .config("reg", 0.005)
    .config("classes", numClasses)
    .config("hidden", 64)
    .function("gcnlayer", "(A,h){Adrop = dropout(A, 0.5); return Adrop@(h@matrix(?, hidden, reg))+vector(?);}")
    .layer("h{l+1}=relu(gcnlayer(A, h{l}))")
    .config("hidden", "classes")  // reassigns the output gcnlayer's "hidden" to be equal to the number of "classes"
    .layer("h{l+1}=gcnlayer(A, h{l})")
    .classify()
    .autosize(new EmptyTensor(numSamples));
```


## Training the model

Training epochs for the created model can be implemented manually, by passing inputs, obtaining outputs, computing losses, and triggering backpropagation on an optimizer. These steps could require lengthy Java code, especially if features like batching or threading parallelization are employed. So, JGNN automates common training patterns by extending a base `ModelTraining` class with training strategies tailored to different data formats and predictive tasks. You can find these subclasses in the [adhoc.train](https://mklab-iti.github.io/JGNN/javadoc/mklab/JGNN/adhoc/train/package-summary.html) package's Javadoc.

Instances of model trainers use a method chain notation to set their parameters. Parameters typically include training and validation data (which should be set first and depend on the model training class) and aspects of the training strategy such as the number of epochs, patience for early stopping, the optimizer used, and loss functions. An example is presented below:

For training, the graph adjacency matrix and node features are already declared as constants by the `FastBuilder` constructor, since node classification takes place on the same graph with fully known node features. Therefore, input features are represented as a column of node identifiers, which the `classify` method uses to gather predictions for respective nodes. Architecture outputs are softmax approximations of the one-hot encodings of respective node labels.

The simplest way to handle missing labels for test data without modifying the example is to leave their one-hot encodings as zeroes only. Additionally, this particular training strategy accepts training and validation data slices, where slices are lists of integer entries pointing to rows of inputs and outputs.

To complete the training setup, the example uses the `Adam` optimization algorithm with a learning rate of _0.01_ and trains over multiple epochs with early stopping. A verbose loss function prints the progress of cross-entropy and accuracy every 10 epochs on the validation data, using cross-entropy for the early stopping criterion. To run a full training process, pass a strategy to a model.

In a cold start scenario, apply a parameter initializer before training begins. A warm start that resumes training from previously trained outcomes would skip this step. Selecting an initializer is not part of the training strategy to emphasize its model-dependent nature; dense layers should maintain the expected input variances in the output before the first epoch, and therefore the initializer depends on the type of activation functions used.

```java
Slice nodes = dataset.samples().getSlice().shuffle(); // A permutation of node identifiers
Matrix inputFeatures = Tensor.fromRange(nodes.size()).asColumn(); // Each node has its identifier as an input (equivalent to: nodes.samplesAsFeatures())
ModelTraining trainer = new SampleClassification()
    // Set training data
    .setFeatures(inputFeatures)
    .setLabels(dataset.labels())
    .setTrainingSamples(nodes.range(0, 0.6))
    .setValidationSamples(nodes.range(0.6, 0.8))
    // Set training strategy
    .setOptimizer(new Adam(0.01))
    .setEpochs(3000)
    .setPatience(100)
    .setLoss(new CategoricalCrossEntropy())
    .setValidationLoss(new VerboseLoss(new CategoricalCrossEntropy(), new Accuracy()).setInterval(10));  // Print every 10 epochs

Model model = modelBuilder.getModel()
    .init(new XavierNormal())
    .train(trainer);
```

## Save and inference

Trained models and their generating builders can be saved and loaded. The next snippet demonstrates how raw predictions can also be made. During this process, some matrix manipulation operations obtain transparent access to parts of the input and output data of the dataset. This access does not copy any data.

```java
modelBuilder.save(Paths.get("gcn_cora.jgnn")); // Needs a Path as an input
Model loadedModel = ModelBuilder.load(Paths.get("gcn_cora.jgnn")).getModel(); // Loading creates a new model builder from which to get the model

Matrix output = loadedModel.predict(Tensor.fromRange(0, nodes.size()).asColumn()).get(0).cast(Matrix.class);
double acc = 0;
for (Long node : nodes.range(0.8, 1)) {
    Matrix nodeLabels = dataset.labels().accessRow(node).asRow();
    Tensor nodeOutput = output.accessRow(node).asRow();
    acc += nodeOutput.argmax() == nodeLabels.argmax() ? 1 : 0;
}
System.out.println("Acc\t " + acc / nodes.range(0.8, 1).size());
```
