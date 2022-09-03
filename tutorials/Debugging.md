# :zap: Debugging
JGNN offers high-level tools for debugging base architectures.
This tutorial covers what errors to expect, what diagnostics to run,
and how to make sense of error messages to fix erroneous architectures.

1. [Name checking](#name-checking)
2. [Debugging execution DAGs](#debugging-execution-dags)
3. [Debugging logical errors](#debugging-logical-errors)
4. [Monitoring operations](#monitoring-operations)

## Name checking
When parsing operations, values should be assigned to variables before
subsequent use. Model builders check for unused variables and raise 
respective runtime exceptions.

For example, for a `GCNBuilder` that tries to parse the expression
`.layer("h{l+1}=relu(hl@matrix(features, 32, reg)+vector(32))")`,
where we remind that the layer definition is an enhanced version of
operation declaration, and `hl` is a typographical error of `h{l}`, 
the following exception is thrown:

```java
Exception in thread "main" java.lang.RuntimeException: Symbol hl not defined.
```

## Debugging execution DAGs
Model builders are responsible for creating directed acyclic graphs (DAGs) 
in models they are managing (these are not to be confused graph inputs GNNs
are management). During parsing, builders may create temporary variables, which 
start with the `_tmp` prefix and are followed by a number, and linking components
to others that use them.

The easiest way to understand execution DAGs is to actually look
at them. The library provides two tools to that end: a) a `.print()`
method for model build functional flows that prints all the parsed
expressions and intermediate expression in the system console, and b)
a. `.getExecutionGraphDot()` that returns a String holding the execution
graph in *.dot* format for visualization with external tools, such
as (GraphViz](https://dreampuf.github.io/GraphvizOnline).

A second error-checking procedure consists of checking 
for model operations that do not
eventually reach any outputs, for example one of the output operation
outcomes defined by `.out(String)`. Avoiding this behavior is particularly
important, as it messes with graph traversal counting during backpropagation.
However, to accomodate complex use cases, these checks can only be manually performed
at the very end of model building with the builder method `.assertBackwardValidity()`.
Calling these checks early on in functional model building
will likely throw exceptions that are not trully logical errors - the
outputs may be declared at later functional steps. Thrown errors would look like this:
```java
Exception in thread "main" java.lang.RuntimeException: The component class mklab.JGNN.nn.operations.Multiply: _tmp102 = null does not lead to an output
	at mklab.JGNN.nn.ModelBuilder.assertBackwardValidity(ModelBuilder.java:504)
	at nodeClassification.APPNP.main(APPNP.java:45)
```
For example, this indicates that the component *_tmp102*  and we should look 
at the execution tre


## Debugging logical errors
There are two main mechanisms for the identification of logically erroneous
architectures: a) mismatched dimension size, and b) mismatched dimension names.
Of the two, dimension sizes are easy to comprehend, since they just mean that
operations are mathematically invalid. 

On the other hand, dimension names need to be determined for
starting data, such as model inputs and parameters, and are automatically
inferred from operations on such primitives. For in-line declaration of
parameters in operations or layers, dimension names are copied from any hyperperameters.
Therefore, for easier debugging, 
prefer using functionl expressions that declare hyperperameters:

```java
new ModelBuilder()
	.config("features", 7)
	.config("hidden", 64)
	.var("x")
	.operation("h = x@matrix(features, hidden)");
```
instead of the simpler `new ModelBuilder().var(x).operation('h = x@matrix(features, hidden)')`


Both mismatched dimensions and mismatched dimension names
throw runtime exceptions. The beginning of their 
error console traces should start with something like this:
```java
java.lang.IllegalArgumentException: Mismatched matrix sizes between SparseMatrix (3327,32) 52523/106464 entries and DenseMatrix (64, classes 6)
During the forward pass of class mklab.JGNN.nn.operations.MatMul: _tmp4 = null with the following inputs:
	class mklab.JGNN.nn.activations.Relu: h1 = SparseMatrix (3327,32) 52523/106464 entries
	class mklab.JGNN.nn.inputs.Parameter: _tmp5 = DenseMatrix (64, classes 6)
java.lang.IllegalArgumentException: Mismatched matrix sizes between SparseMatrix (3327,32) 52523/106464 entries and DenseMatrix (64, classes 6)
	at mklab.JGNN.core.Matrix.matmul(Matrix.java:258)
	at mklab.JGNN.nn.operations.MatMul.forward(MatMul.java:21)
	at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:180)
	at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:170)
	at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:170)
	at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:170)
	at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:170)
	at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:170)
	...
```

As an example, let us try to understand what this error tels us. First, 
it notifies us of the actual problem: that the architecture encounters mismatched matrix
sizes when trying to multiply a 3327x32 SparseMatrix with a 64x6 dense matrix. 
This is easy to understand and there are also dimension names in there;
for this example, only *classes* is a named dimension, but if models
and input data are well-designed more names will be in there and some
errors will also arise from different dimension names.

At any rate, understanding the exact error is easy - the inner matrix dimensions
of matrix multiplication
do not agree. However, we need to find the error within our architecture to 
be able to fix whatever is causing this.

To do this, we continue reading and see the message
`During the forward pass of class mklab.JGNN.nn.operations.MatMul: _tmp4 = null`.
This tells us that the problem occurs when trying to calculate *_tmp4*
which currently is currently assigned a *null* tensor as value (this is pretty normal,
as the forward pass has not yet already concluded for that variable to assume a value).
Some more information is there to see what the operation's inputs are like - in this case
they coincide with the multiplication's inputs, but this will not always be the case.

The important point, is to go back to the execution tree and see during which exact operation
this variable is defined. There, we will undoubtedly find that some dimension had 64 instead
of 32 elements or conversely.

## Monitoring operations
In addition to all other debugging mechanisms, JGNN presents a way to view when
forward and backward operations of specific code components are executed and with what kinds
of arguments.
This can be particularly useful when testing new components in real (complex) architectures.

The practice consists of calling a *monitor(...)* function within operations.
This does not affect what expressions do and only enables printing execution tree operations
on operation components. For example, to monitor the outcome of matrix multiplication within 
the following operation:

```java
builder.operation("h = relu(x@matrix(features, 64) + vector(64))")
```

it should be converted to:

```java
builder.operation("h = relu(monitor(x@matrix(features, 64)) + vector(64))")
```