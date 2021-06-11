# Primitives
JGNN provides two classes of primitives written in native java: vectors and matrices.

## Vector operations
JGNN provides vector operations through a `mklab.JGNN.core.Tensor` abstract class. This class implements four types of methods:

**- Basic arithmetic operations.** 
These create new tensors using the abstract `zeroCopy()` method to create a vector of zeroes and include methods such as `Tensor add(Tensor), Tensor substract(Tensor), Tensor multiply(Tensor), Tensor multiply(double)`, `Tensor normalized(), Tensor toProbability()`. Multiplication is performed element-by-element.

**- In-place arithmetic operations.** 
These always return the object itself and either begin with a "self" prefix, such as `Tensor selfAdd(Tensor), Tensor selfSubtract(Tensor), Tensor setMultiply(Tensor), Tensor selfMultiply(double double)`, or a "setTo" prefix if they directly transform the tensor, such as `Tensor setToZero(), Tensor setToRandom(), Tensor setToOnes(), Tensor setToNormalized(), Tensor setToProbability()`. Prefer in-place arithmetic operations when transforming tensor values, as these do not allocate new memory. For example, the following code can be used for fast normalization of a tensor of ones without using additional memory:

```Java
Tensor normalized = new DenseTensor(10).setToOnes().setToNormalized()
```

**- Basic calculations.** These include the dot product `double dot(Tensor)` as well as summary statistics `double norm(), double max(), double min(), double sum()`.

**- Element access.** Tensor elements can be accessed and changed through the `Tensor set(long position, double value)` and `double get(long position)` values. Setting to NaN values purposefully throws exceptions. Make sure that iterating over non-zero elements is performed through the iterator `Iterator<Long> getNonZeroElements()`; this basically traverses all elements for dense tensors, but skips zero elements for sparse tensors.

## Vector initialization

You can initialize either a dense tensor with the expression `Tensor denseTensor = new mklab.JGNN.tensor.DenseTensor(long size)` or, if there are many zero elements expected, a sparse tensor per `Tensor sparseTensor = new mklab.JGNN.tensor.SparseTensor(long size)`. For example, an one-hot encoding of the fifth out of ten classes can be performed per `Tensor encodingSecond = new mklab.JGNN.tensor.DenseTensor(10).set(4, 1)`.

Dense tensors can be serialized with their `String toString()` method and deserialized with a respective constructor `mklab.JGNN.tensor.DenseTensor(String)`.


## Matrices
To be completed.