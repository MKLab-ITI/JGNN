# Debugging

JGNN offers high-level tools for debugging architectures. Here we cover what diagnostics to run, and how to make sense of error messages to fix erroneous architectures. 

## Missing symbols

We already mentioned that model builder symbols should be assigned before subsequent use. For example, consider a `FastBuilder` that tries to parse the expression `.layer("h{l+1}=relu(hl@matrix(features, 32, reg)+vector(32))")`, where `hl` is a typographical error of `h{l}`. In this case, an exception is thrown:  
`Exception in thread "main" java.lang.RuntimeException: Symbol hl not defined.`

Internally, models are effectively directed acyclic graphs (DAGs) that model builders create. DAGs should not be confused with the graphs that GNNs architectures analyze; they are just an organization of data flow between `NNComponent`s. During parsing, builders may create temporary variables, which start with the `_tmp` prefix and are followed by a number. Temporary variables often link components to others that use them. The easiest way to understand execution DAGs is to look at them. The library provides two tools for this purpose: a `.print()` method that prints built functional flows in the system console, and a `.getExecutionGraphDot()` method that returns a string holding the execution graph in *.dot* format for visualization with tools like [GraphViz](https://dreampuf.github.io/GraphvizOnline).

Another error-checking procedure consists of an assertion that all model operations eventually affect at least one output. Computational branches that lead nowhere mess up the DAG traversal during backpropagation and should be checked with the method `.assertBackwardValidity()`. The latter throws an exception if an invalid model is found. Performing this assertion early on in model building will likely throw exceptions that are not logical errors, given that independent outputs may be combined later. Backward validity errors look like this in the following example. This indicates that the component `_tmp102` does not lead to an output, and we should look at the execution tree to understand its role.

```text
Exception in thread "main" java.lang.RuntimeException: The component class mklab.JGNN.nn.operations.Multiply: _tmp102 = null does not lead to an output
at mklab.JGNN.nn.ModelBuilder.assertBackwardValidity(ModelBuilder.java:504)
at nodeClassification.APPNP.main(APPNP.java:45)
```

## Mismatched dimensions

Some tensor or matrix methods do not correspond to numerical operations but are only responsible for naming dimensions. Functionally, such methods are largely decorative, but they can improve debugging by throwing errors for incompatible non-null names. For example, adding two matrices with different dimension names will result in an error. Likewise, the inner dimension names during matrix multiplication should agree. Arithmetic operations, *including* matrix multiplication and copying, automatically infer dimension names in the result to ensure that only compatible data types are compared. Dimension name changes do *not* backtrack the changes, even for see-through data types, such as the outcome of `asTransposed()`. Matrices effectively have three dimension names: for their rows, columns, and inner data as long as they are treated as tensors.

| Operation | Comments |
|-----------|----------|
| `Tensor setDimensionName(String name)` | For naming tensor dimensions (of the 1D space tensors lie in). |
| `Tensor setRowName(String rowName)` | For naming what kind of information matrix rows hold (e.g., "samples"). Defined only to matrices. |
| `Tensor setColName(String colName)` | For naming what kind of information matrix columns hold (e.g., "features"). Defined only for matrices. |
| `Tensor setDimensionName(String rowName, String colName)` | A shorthand of calling `setRowName(rowName).setColName(colName)`. Defined only for matrices. |

There are two main mechanisms for identifying logical errors within architectures: a) mismatched dimension size, and b) mismatched dimension names. Of the two, dimension sizes are easier to comprehend since they just mean that operations are mathematically invalid. On the other hand, dimension names need to be determined for starting data, such as model inputs and parameters, and are automatically inferred from operations on such primitives. For in-line declaration of parameters in operations or layers, dimension names are copied from any hyperparameters. Therefore, for easier debugging, prefer using functional expressions that declare hyperparameters, like below.

```java
new ModelBuilder()
    .config("features", 7)
    .config("hidden", 64)
    .var("x")
    .operation("h = x@matrix(features, hidden)");
```

Both mismatched dimensions and mismatched dimension names throw runtime exceptions. The beginning of their error console traces should start with something like this:

```text
java.lang.IllegalArgumentException: Mismatched matrix sizes between SparseMatrix (3327,32) 52523/106464 entries and DenseMatrix (64, classes 6)
During the forward pass of class mklab.JGNN.nn.operations.MatMul: _tmp4 = null with the following inputs:
class mklab.JGNN.nn.activations.Relu: h1 = SparseMatrix (3327,32) 52523/106464 entries
class mklab.JGNN.nn.inputs.Parameter: _tmp5 = DenseMatrix (64, classes 6)
java.lang.IllegalArgumentException: Mismatched matrix sizes between SparseMatrix (3327,32) 52523/106464 entries and DenseMatrix (64, classes 6)
at mklab.JGNN.core.Matrix.matmul(Matrix.java:258)
at mklab.JGNN.nn.operations.MatMul.forward(MatMul.java:21)
at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:180)
at mklab.JGNN.nn.NNOperation.runPrediction(NNOperation.java:170)
...
```

This particular stack trace tells us that the architecture encounters mismatched matrix sizes when trying to multiply a 3327x32 `SparseMatrix` with a 64x6 `DenseMatrix`. Understanding the exact error is straightforward—the inner dimensions of matrix multiplication do not agree. However, to fix the error within our architecture, we need to find out where it occurs. The error message indicates:

`During the forward pass of class mklab.JGNN.nn.operations.MatMul: _tmp4 = null`

This tells us that the problem occurs when trying to calculate `_tmp4`, which is currently assigned a `null` tensor as its value. Additional information is provided to show what the operation's inputs look like—in this case, they coincide with the multiplication's inputs, but this will not always be the case. The important point is to go back to the execution tree and see during which exact operation this variable is defined. There, we will likely find that some dimension had 64 instead of 32 elements or vice versa.


## Check for architecture validity

In addition to all other debugging mechanisms, JGNN provides a way to show when forward and backward operations of specific code components are executed and with what kinds of arguments. This can be particularly useful when testing new components in real (complex) architectures. The practice consists of calling a `monitor(...)` function within operations. This does not affect what expressions do and only enables printing execution tree operations on operation components. For example, the next snippet monitors the outcome of matrix multiplication:

```java
builder.operation("h = relu(monitor(x@matrix(features, 64)) + vector(64))");
```