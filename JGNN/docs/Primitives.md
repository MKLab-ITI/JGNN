# Primitives
JGNN provides a `mklab.JGNN.core.Tensor` abstract class for storing calculation primitives. Vector and matrix operations use primitives of this or derived classes. To reduce the number of terms and improve comprehensibility of source code, operations between two tensors are implemented by calling respective methods of the former. 

## Tensor operations
Tensor operations are performed element-by-element and can be split into basic arithmetic, in-place arithmetic, summary statistic and element access ones. Basic arithmetics combine the values of two tensors to create a new one, whereas in-place arithmetics affect the tensor's values and begin with with a "self" prefix for pairwise operations or "setTo" prefix to perform operators. Summary statistics comprise operations that output sinmple numeric values. Finally, element access allows manipulation of specific values. Here we present some commonly used operations applicable to all tensors, whose functionality is inferable from their name and argument types. For more operations or details, please refer to the project's Javadoc.

- Example of basic arithmetic operations: `Tensor zeroCopy(),Tensor add(Tensor), Tensor substract(Tensor), Tensor multiply(Tensor), Tensor multiply(double)`, `Tensor normalized(), Tensor toProbability()`. Zero copies share the same type with the tensor and comprise only zeros. Multiplication is performed element-by-element.

- Example of in-place arithmetic operations: `Tensor selfAdd(Tensor), Tensor selfSubtract(Tensor), Tensor setMultiply(Tensor), Tensor selfMultiply(double double), Tensor setToZero(), Tensor setToRandom(), Tensor setToOnes(), Tensor setToNormalized(), Tensor setToProbability()`. 

- Example of summary statistics operations: `double dot(Tensor), double norm(), double max(), double min(), double sum()`

- Element access operations: Tensor elements can be accessed and changed through the `Tensor set(long position, double value)` and `double get(long position)` values. Setting to NaN values purposefully throws exceptions. `Iterator<Long> getNonZeroElements()` traverses all elements for dense tensors, but skips zero elements for sparse tensors.


:bulb: Prefer in-place arithmetic operations when transforming tensor values or when tensors are stored as an intermediate calculation step, as these do not allocate new memory. For example, the following code can be used for fast normalization of a tensor of ones without using additional memory.

:bulb: To write code that accommodates both dense and sparse tensors, make sure that iterating over non-zero elements is performed through the iterator `Iterator<Long> getNonZeroElements()`


```Java
Tensor normalized = new DenseTensor(10).setToOnes().setToNormalized()
```


## Vector initialization

You can initialize either a dense tensor with the expression `Tensor denseTensor = new mklab.JGNN.tensor.DenseTensor(long size)` .
If there are many zero elements expected or if sizes go beyond the max integer limit Java imposes on array sizes (and hence a dense representation can not be stored as an array), a sparse tensor can be used per `Tensor sparseTensor = new mklab.JGNN.tensor.SparseTensor(long size)`. For example, one-hot encodings for classification problems can be generated with the following code, which creates a dense tensor with $numClasses$ elements and puts at place $classId \in\{0,1,..numClasses-1\}$ the value 1:

```java
int classId = ...;
int numClasses = ...;
Tensor oneHotEncoding = new mklab.JGNN.tensor.DenseTensor(numClasses).set(classId, 1);
```

Dense tensors can be serialized with their `String toString()` method and deserialized with a respective constructor `mklab.JGNN.tensor.DenseTensor(String)`.


## Matrices
To be completed.