# Primitives
Primitive operations found in other tutorials suffice for most 
use cases. However, you may need to process
neural inputs, postprocess learning outcomes, create custom 
parameters, contribute to the library with more components,
or make derivative works based on native java vector and matrix
arithmetics.

This tutorial lists JGNN primitives and explains how to work with them.

In general, JGNN provides the `mklab.JGNN.core.Tensor` abstract class 
for storing data. Vector and matrix operations use primitives 
of this or derived classes. To reduce the number of code predicates
and improve comprehensibility, operations between two tensors are implemented 
by calling respective methods of the first one. 

## Table of contents

1. [Tensor operations](#tensor-operations)
2. [Vector initialization](#vector-initialization)
3. [Matrix initialization and operations](#matrix-initialization-and-operations)
4. [Named dimensions](#named-dimensions)

## Tensor operations
Tensor operations are performed element-by-element and can be split into
the following categories:
<br>a) *arithmetic* - combine the values of two tensors to create a new one
<br>b) *in-place arithmetic* - combine the values of two tensors to alter the first one
<br>c) *summary statistics* - output simple numeric values
<br>c) *element access* - manipulation of specific values


:bulb: In-place arithmetics follow the same naming conventions of base arithmetics and
begin with with a "self" prefix for pairwise operations or "setTo" prefix to perform operators.

Here we present some commonly used operations applicable to all tensors, whose functionality is inferable
from their name and argument types. For more operations or details, please refer to the project's 
[Javadoc](https://mklab-iti.github.io/JGNN/).

Operation | Type | Comments
--- | --- | ---
`Tensor copy()` | arithmetic
`Tensor zeroCopy()` | arithmetic | Zero copies share the same type with the tensor and comprise only zeros. 
`Tensor add(Tensor)` | arithmetic
`Tensor substract(Tensor)` | arithmetic 
`Tensor multiply(Tensor)` | arithmetic | Multiplication is performed element-by-element.
`Tensor multiply(double)` | arithmetic
`Tensor normalized()` | arithmetic | Division with L2 norm (if non-zero).
`Tensor toProbability()` | arithmetic | Division with the sum (if non-zero).
`Tensor setToZero()` | in-place arithmetic 
`Tensor selfAdd(Tensor)` | in-place arithmetic
`Tensor selfSubtract(Tensor)` | in-place arithmetic
`Tensor setMultiply(Tensor)` | in-place arithmetic
`Tensor selfMultiply(double)` | in-place arithmetic
`Tensor setToRandom()` | in-place arithmetic | element selected from uniform distribution in the range [0,1]
`Tensor setToOnes()` | in-place arithmetic
`Tensor setToNormalized()` | in-place arithmetic  | Division with L2 norm (if non-zero).
`Tensor setToProbability()` | in-place arithmetic  | Division with the sum (if non-zero).
`double dot(Tensor)` | summary statistics
`double norm()` | summary statistics | The L2 norm.
`double sum()` | summary statistics
`double max()` | summary statistics
`double min()` | summary statistics
`long argmax()` | summary statistics
`long argmin()` | summary statistics
`double toDouble()` | summary statistics | Converts tensor with exactly one element to a double (throws exception if more elements).
`Tensor set(long position, double value)` | element access | NaN values throw exceptions. Is in-place.
`double get(long position)` | element access
`Iterator<Long> getNonZeroElements()` | element access | Traverses all elements for dense tensors, but skips zero elements for sparse tensors. (Guarantee: there is no non-zero element not traversed.) Returns element positions **positions**.
`String describe()` | summary statistics | Description of type and dimensions.


:bulb: To write code that accommodates both dense and sparse tensors, make sure that iterating over indices elements is performed with the iterator `Iterator<Long> getNonZeroElements()`.

Prefer in-place arithmetic operations when transforming tensor values or for intermediate calculation steps, as these do not allocate new memory. For example, the following code can be used for creating and normalizing a tensor of ones without using additional memory:

```Java
Tensor normalized = new DenseTensor(10).setToOnes().setToNormalized();
```


## Vector initialization

You can initialize a dense tensor with the expression `Tensor denseTensor = new mklab.JGNN.tensor.DenseTensor(long size)` .
If there are many zero elements expected, or if sizes go beyond the max integer limit Java imposes on array sizes (and hence a dense representation can not be stored as an array), a sparse tensor can be used per `Tensor sparseTensor = new mklab.JGNN.tensor.SparseTensor(long size)`. For example, one-hot encodings for classification problems can be generated with the following code. This creates a dense tensor with *numClasses* elements and puts at element *classId* the value 1:

```java
int classId = ...;
int numClasses = ...;
Tensor oneHotEncoding = new mklab.JGNN.tensor.DenseTensor(numClasses).set(classId, 1);
```

Dense tensors serialized with their `String toString()` method and can be deserialized into new tensors with the constructor `mklab.JGNN.tensor.DenseTensor(String)`.


## Matrix initialization and operations
The `Matrix` class extends the concept of tensors with additional operations. Under the hood,
Matrices linearly store elements and use computations to transform the (row,col) position of
their elements to respective positions. The outcome of some methods inherited from tensors may
need to be typecast back into a matrix (e.g. for all in-place operations).

Operation | Type | Comments
--- | --- | ---
`Matrix onesMask()` | arithmetic | Copy of a matrix with elements set to one.
`Matrix transposed()` | arithmetic | There is no method for in-place transposition.
`Matrix asTransposed()` | arithmetic | Shares data with the original.
`Tensor getRow(long)` | arithmetic | Shares data with the original.
`Tensor getCol(long)` | arithmetic | Shares data with the original.
`Tensor transform(Tensor x)` | arithmetic | Outputs a dense tensor that holds the linear transformation of the given tensor (using it as a column vector) by multiplying it with the matrix.
`Matrix matmul(Matrix with)` | arithmetic | Outputs the matrix multiplication **this \* with**. There is no in-place matrix multiplication.
`Matrix matmul(Matrix with, boolean transposeSelf, boolean transposeWith)` | arithmetic | Does not perform memory allocation to compute transpositions.
`Matrix external(Tensor horizontal, Tensor vertical)` | static method | External product of two tensors. Is a dense matrix.
`Matrix symmetricNormalization()` | in-place arithmetic | The symmetrically normalized matrix.
`Matrix setToSymmetricNormalization()` | in-place arithmetic | The symmetrically normalized matrix.
`Matrix setMainDiagonal(double value)` | in-place arithmetic | Sets diagonal elements.
`Matrix setDiagonal(long diagonal, double value)` | in-place arithmetic | Sets diagonal elements.
`Matrix put(long row, long col, double value)` | element access | NaN values throw exceptions. Is in-place.
`Iterable<Entry<Long, Long>> getNonZeroEntries()` | element access | Similar to getNonZeroElements() but iterates through (row, col) pairs.


## Named dimensions
In addition to other operations, there exist methods that do not affect
tensor or matrix values but are only responsible for naming dimensions. Functioanlly, these
are decorative and  aim to improve debugging by throwing errors for incompatible non-null names. 
For example, adding two matrices with different dimension names will result in an error. 
Likewise, the inner dimension names during matrix multiplication should agree.

Operation | Type | Comments
--- | --- | ---
`Tensor setDimensionName(String name)` | arithmetic | For naming tensor dimensions (of the 1D space tensors lie in).
 `Tensor setRowName(String rowName)` | arithmetic | For naming what kind of information matrix rows hold (e.g. `"samples"`).
 `Tensor setColName(String colName)` | arithmetic | For naming what kind of information matrix columns hold (e.g. `"features"`).
 `Tensor setDimensionName(String rowName, String colName)` | arithmetic | A shorthand of calling `setRowName(rowName).setColName(colName)`.

 
Arithmetic operations, *including* matrix multiplication and copying,
automatically infer dimension names in the result to make sure that only compatible data types 
are compared. Dimension names can be freely changed for any Tensor *without*
backtracking changes (even for see-through data types, such as the outcome of asTransposed()).

:bulb: Matrices effectively have three dimension names: for their rows, columns, and inner
data as long as they are treated as tensors.
 
 
[NEXT: Debugging](Debugging.md)
