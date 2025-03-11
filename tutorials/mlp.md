# Traditional classification

<i>by [Emmanouil Krasanakis](https://github.com/maniospas).</i>

## Setup

For this tutorial, there are no additional dependencies. JGNN 1.3.40 or later is needed.

## Loading data

We will use the Citesee dataset, where the goal
is to classify papers into different categories. In the literature, citations between the
papers may be used to improve the classification, but here we show only how some given textual
features can be used to create the classification. We obtain those data in matrix format like
so and print some characteristics over the number of samples and label feature matrix sizes.
For other tasks, construct similar matrices. Labels contain one-hot encodings for corresponding samples.

```java
Dataset dataset = new Citeseer();

System.out.println("Samples\t: " + dataset.samples().size());
System.out.println("Labels\t: " + dataset.labels().describe());
System.out.println("Features: " + dataset.features().describe());
```

```cpp
Samples : 3327
Labels  : SparseMatrix (3327,6) 3327/19962 entries
Features: SparseMatrix (3327,3703) 105165/12319881 entries
```


## Arcitecture

Continuing the example, we define a two-layer perceptron for classification using
a model builder to parse Neuralang expressions. Neuralang is a declarative scripting
language provided by JGNN to simplify otherwise complex architecture declarations.
The built architecture has only one hidden layer and softmax activation on its outputs
to mirro that it is aiming to replicate one-hot distrivutions. Chosen activation is the
relu.

The architecture sets several configuration hypterparameters to control the number of
inputs, outputs, hidden dimensions, and regularization.

```java
long numFeatures = dataset.features().getCols();
long numClasses = dataset.labels().getCols();
ModelBuilder modelBuilder = new ModelBuilder()
        .config("features", numFeatures)
        .config("classes", numClasses)
        .config("hidden", 64)
        .config("regularization", 0.005)
        .var("x")
        .operation("h = relu(x@matrix(features, classes, regularization)+vector(classes))")
        .operation("yhat = softmax(h@matrix(classes, classes)+vector(classes), dim: 'row')")
        .out("yhat")
        .print();

System.out.println(modelBuilder.getExecutionGraphDot());
```


## Training scheme

We next create a training scheme. This uses the Adam optimizer with a learning rate of
0.01 and trains for a maximum of 3000 epochs with early stopping; patience of 100
means that training stops if the validation metric does not improve for that many epochs.
The loss function is binary cross entropy, and validation accuracy is tracked and
printed every 10 epochs. We do not wrap the optimizer within a regulariation strategy because,
a penalty is added internally by the architecture when computing gradients.

Samples get a 60-20-20 split between training-validation-testing data, where validation
data are the ones determining early stopping based on accuracy. At the same time,
a large number of data batches are generated. These are mutually exclusive for this training
scheme and we also set them up for parallel execution. This speeds up computations
significantly in multicore CPUs - try with 1 batch to see the difference in training
times. Note that validation computations run in the same batch, so you can choose a smaller
validation set to get further speedups.

```java
Slice samples = dataset.samples().getSlice().shuffle(100);
ModelTraining trainer = new SampleClassification()
        .setFeatures(dataset.features())
        .setOutputs(dataset.labels())
        .setTrainingSamples(samples.range(0, 0.6))
        .setValidationSamples(samples.range(0.6, 0.8))
        .setOptimizer(new Adam(0.01))
        .setEpochs(3000)
        .setPatience(100)
        .setNumParallelBatches(20)
        .setLoss(new BinaryCrossEntropy())
        .setValidationLoss(new VerboseLoss(new Accuracy()).setInterval(10));
```

## Results

The model is initialized with XavierNormal and trained using the defined trainer.
We evaluate the final model on the test set (20% of the dataset). Evaluation takes
advantage of JGNN primitive operations to access only specific segments of the
data without copying anything.

```java
long tic = System.currentTimeMillis();
Model model = modelBuilder.getModel()
        .init(new XavierNormal())
        .train(trainer);
long toc = System.currentTimeMillis();

double acc = 0;
for(Long node : nodeIds.range(0.8, 1)) {
    Matrix nodeFeatures = dataset.features().accessRow(node).asRow();
    Matrix nodeLabels = dataset.labels().accessRow(node).asRow();
    Tensor output = model.predict(nodeFeatures).get(0);
    acc += output.argmax()==nodeLabels.argmax()?1:0;
}
System.out.println("Acc\t "+acc/nodeIds.range(0.8, 1).size());
System.out.println("Time\t "+(toc-tic)/1000.);
```

```cpp
Epoch 0  Accuracy 0.192
Epoch 10  Accuracy 0.238
Epoch 20  Accuracy 0.293
Epoch 30  Accuracy 0.286
Epoch 40  Accuracy 0.289
Epoch 50  Accuracy 0.301
Epoch 60  Accuracy 0.328
Epoch 70  Accuracy 0.347
Epoch 80  Accuracy 0.364
Epoch 90  Accuracy 0.391
Epoch 100  Accuracy 0.364
Epoch 110  Accuracy 0.409
Epoch 120  Accuracy 0.292
Epoch 130  Accuracy 0.352
Epoch 140  Accuracy 0.409
Epoch 150  Accuracy 0.439
Epoch 160  Accuracy 0.421
Epoch 170  Accuracy 0.472
Epoch 180  Accuracy 0.471
Epoch 190  Accuracy 0.493
Epoch 200  Accuracy 0.538
Epoch 210  Accuracy 0.421
Epoch 220  Accuracy 0.495
Epoch 230  Accuracy 0.538
Epoch 240  Accuracy 0.516
Epoch 250  Accuracy 0.582
Epoch 260  Accuracy 0.49
Epoch 270  Accuracy 0.534
Epoch 280  Accuracy 0.486
Epoch 290  Accuracy 0.519
Epoch 300  Accuracy 0.597
Epoch 310  Accuracy 0.57
Epoch 320  Accuracy 0.609
Epoch 330  Accuracy 0.556
Epoch 340  Accuracy 0.562
Epoch 350  Accuracy 0.561
Epoch 360  Accuracy 0.552
Epoch 370  Accuracy 0.561
Epoch 380  Accuracy 0.582
Epoch 390  Accuracy 0.606
Epoch 400  Accuracy 0.627
Epoch 410  Accuracy 0.62
Epoch 420  Accuracy 0.642
Epoch 430  Accuracy 0.531
Epoch 440  Accuracy 0.624
Epoch 450  Accuracy 0.589
Epoch 460  Accuracy 0.626
Epoch 470  Accuracy 0.618
Epoch 480  Accuracy 0.635
Epoch 490  Accuracy 0.611
Epoch 500  Accuracy 0.638
Acc	 0.6291291291291291
Time	 91.847
```
