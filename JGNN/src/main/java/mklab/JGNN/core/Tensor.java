package mklab.JGNN.core;

import java.util.Iterator;

import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.core.tensor.AccessSubtensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Range;

/**
 * This class provides a native java implementation of Tensor functionalities.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class Tensor implements Iterable<Long> {
	private long size;
	private String dimensionName;
	
	/**
	 * Construct that creates a tensor of zeros given its number of elements
	 * @param size The number of tensor elements
	 */
	public Tensor(long size) {
		init(size);
	}
	protected Tensor() {}
	
	/**
	 * Sets a name for the tensor's one dimension. If set, names are checked for
	 * compatibility during operations, so that tensors laying across different dimensions
	 * do not match. Removed dimension names are matched to anything.
	 * @param dimensionName The new row name or <code>null</code> to remove current name.
	 * @return <code>this</code> Tensor instance.
	 * @see #getDimensionName()
	 */
	public final Tensor setDimensionName(String dimensionName) {
		this.dimensionName = dimensionName;
		return this;
	}
	public final String getDimensionName() {
		return dimensionName;
	}
	/**
	 * Set tensor elements to random values from the uniform range [0,1]
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToRandom() {
		for(long i=0;i<size();i++)
			put(i, Math.random());
		return this;
	}
	/**
	 * Set tensor elements to random values by sampling them from a given {@link Distribution}
	 * instance.
	 * @param distribution The distribution instance to sample from.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToRandom(Distribution distribution) {
		for(long i=0;i<size();i++)
			put(i, distribution.sample());
		return this;
	}
	/**
	 * Can be called instead of <code>super(size)</code> by inheriting class
	 * constructors as needed after. Sets the tensor {@link #size()} to the given value
	 * and calls the {@link #allocate(long)} method.
	 * @param size The size of the tensor.
	 */
	protected final void init(long size) {
		this.size = size;
		allocate(size);
	}
	/**
	 * Asserts that the tensor holds only finite values. Helps catch errors
	 * early on and avoid misidentifying models as high quality by comparing
	 * desired outcomes with NaN when in reality they pass through infinity and hence
	 * don't converge.
	 * @throws RuntimeException if one or more tensor elements are NaN or Inf.
	 */
	public void assertFinite() {
		for(long i : getNonZeroElements())
			if(!Double.isFinite(get(i)))
				throw new RuntimeException("Did not find a finite value");
	}
	protected abstract void allocate(long size);
	/**
	 * If the subclassed tensor allows it, release all memory it takes up
	 * so that the garbage collector will eventually clean it up. This
	 * memory will be released anyway by Java once there are no more
	 * references to the object.
	 * @see #persist()
	 * @deprecated This method may not be present in future versions
	 *  of the library, depending on whether memory reuse proves useful or nor.
	 */
	public abstract void release();
	/**
	 * If supported by the subclassed tensor, invalidates calls to
	 * {@link #release()} so that memory is a de-allocated only when
	 * object references expire.
	 * @see #release()
	 * @deprecated This method may not be present in future versions
	 *  of the library, depending on whether memory reuse proves useful or nor.
	 */
	public abstract void persist();
	/**
	 * Assign a value to a tensor element. All tensor operations use this function to wrap
	 * element assignments.
	 * @param pos The position of the tensor element
	 * @param value The value to assign
	 * @throws RuntimeException If the value is NaN or the element position is less than 0 or greater than {@link #size()}-1.
	 * @return <code>this</code> Tensor instance.
	 */
	public abstract Tensor put(long pos, double value);
	/**
	 * Retrieves the value of a tensor element at a given position. All tensor operations use this function to wrap
	 * element retrieval.
	 * @param pos The position of the tensor element
	 * @return The value of the tensor element
	 * @throws RuntimeException If the element position is less than 0 or greater than {@link #size()}-1.
	 */
	public abstract double get(long pos);
	/**
	 * Add a value to a tensor element.
	 * @param pos The position of the tensor element
	 * @param value The value to assign
	 * @return <code>this</code> Tensor instance.
	 * @see #put(long, double)
	 */
	public final Tensor putAdd(long pos, double value) {
		put(pos, get(pos)+value);
		return this;
	}
	/**
	 * @return The number of tensor elements
	 */
	public final long size() {
		return size;
	}
	/**
	 * Asserts that the tensor's {@link #size()} matches the given size.
	 * @param size The size the tensor should match
	 * @throws RuntimeException if the tensor does not match the given size
	 */
	public final void assertSize(long size) {
		if(size()!=size)
			throw new RuntimeException("Different sizes: given "+size+" vs "+size());
	}
	/**
	 * Asserts that the tensor's dimensions match with another tensor. This check can be made
	 * more complex by derived classes, but for a base Tensor instance it is equivalent {@link #assertSize(long)}.
	 * This method calls {@link #isMatching(Tensor)} to compare the tensors and throws an exception
	 * if it returns false.
	 * @param other The other tensor to compare with.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor assertMatching(Tensor other) {
		if(!isMatching(other)) 
			throw new RuntimeException("Non-compliant: "+describe()+" vs "+other.describe());
		return this;
	}
	/**
	 * Compares the tensor with another given tensor to see if they are of a same type.
	 * In the simplest (and weakest) case this checks whether the {@link #size()} is the same,
	 * but in more complex cases, this may check additional properties, such as a matching number
	 * of rows and columns in matrices.
	 * @param other The tensor to compare to.
	 * @return Whether binary operations can be performed between the given tensors given
	 * their characteristics (e.g. type, size, etc.).
	 * @see #assertMatching(Tensor)
	 */
	protected boolean isMatching(Tensor other) {
		return size==other.size() && (dimensionName==null || other.getDimensionName()==null || dimensionName==other.getDimensionName());
	}
	/**
	 * Creates a tensor of the same class with the same size and all element set to zero.
	 * @return A tensor with the same size.
	 * @see #zeroCopy(long)
	 */
	public Tensor zeroCopy() {
		return zeroCopy(size()).setDimensionName(this);
	}
	/**
	 * Fills in dimension names per an example {@link isMatching} tensor. This appropriately fills in dimension
	 * names of inherited classes too, such as matrices. Effectively, this method automatically infers
	 * dimension names during operations.
	 * @param other The tensor from which to retrieve dimension names.
	 * @return <code>this</code> Tensor instance.
	 */
	public Tensor setDimensionName(Tensor other) {
		assertMatching(other); 
		if(dimensionName==null)
			dimensionName = other.getDimensionName();
		return this;
	}
	/**
	 * Creates a tensor of the same class with  a given size and all element set to zero.
	 * @param size The size of the new tensor.
	 * @return A new tensor.
	 * @see #zeroCopy(long)
	 */
	public abstract Tensor zeroCopy(long size);
	
	@Override
	public Iterator<Long> iterator() {
		return traverseNonZeroElements();
	}
	/**
	 * Retrieves an iterable that wraps {@link #traverseNonZeroElements()}.
	 * For the time being, <code>this</code> is returned by implementing Iterable,
	 * but this only serves the practical purpose of avoiding to instantiate
	 * a new object in case many tensors are used.
	 * @return An iterable of tensor positions.
	 */
	public final Iterable<Long> getNonZeroElements() {
		return this;
	}
	/**
	 * Provides an estimation for the non-zero number of elements stored in the tensor,
	 * where this number is equal to the size for dense tensors, but equal to the actual
	 * number of non-zero elements for sparse tensors.
	 * Basically, this quantity is proportional to the allocated memory.
	 * @return A long number equal to or less to the tensor size.
	 * @see #density()
	 */
	public long estimateNumNonZeroElements() {
		/*long ret = 0;
		for(long pos : getNonZeroElements())
			if(get(pos)!=0)
				ret += 1;
		return ret;*/
		return size;
	}
	
	/**
	 * Provides the memory allocation density of {@link #getNonZeroElements()}
	 * compare to the size of the tensor. 1 indicates fully dense tensors,
	 * and lower values sparser data.
	 * @return A double in the range [0,1].
	 */
	public final double density() {
		return estimateNumNonZeroElements() / (double)size;
	}
	
	/**
	 * Retrieves positions within the tensor that may hold non-zero elements.
	 * This guarantees that <b>all non-zero elements positions are traversed</b> 
	 * but <b>some of the returned positions could hold zero elements</b>.
	 * For example, {@link mklab.JGNN.core.tensor.DenseTensor} traverses all
	 * of its elements this way, whereas {@link mklab.JGNN.core.tensor.SparseTensor}
	 * indeed traverses only non-zero elements.
	 * 
	 * @return An iterator that traverses positions within the tensor.
	 */
	public abstract Iterator<Long> traverseNonZeroElements();
	/**
	 * Creates a {@link #zeroCopy()} and transfers to it all potentially non-zero element values.
	 * @return a copy of the Tensor with the same size and contents
	 * @see #zeroCopy()
	 * @see #getNonZeroElements()
	 */
	public final Tensor copy() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, get(i));
		return res;
	}
	/**
	 * Performs a sparse assignment.
	 * @param tensor The tensor whose elements to copy (it's not affected).
	 * @return <code>this</code> Tensor instance.
	 * @see #assign(Tensor)
	 */
	public final Tensor assign(Tensor tensor) {
		assertMatching(tensor);
		for(long i : tensor.getNonZeroElements())
			put(i, tensor.get(i));
		return this;
	}
	/**
	 * @param tensor The tensor to add with
	 * @return a new Tensor that stores the outcome of addition
	 */
	public final Tensor add(Tensor tensor) {
		assertMatching(tensor);
		if(density()<tensor.density()) 
			return tensor.add(this);
		Tensor res = copy();
		for(long i : tensor.getNonZeroElements())
			res.put(i, get(i)+tensor.get(i));
		return res;
	}
	/**
	 * @param value The value to add to each element
	 * @return a new Tensor that stores the outcome of addition
	 */
	public final Tensor add(double value) {
		Tensor res = zeroCopy();
		for(long i=0;i<size();i++)
			res.put(i, get(i)+value);
		return res;
	}
	/**
	 * Performs in-memory addition to the Tensor, storing the result in itself.
	 * @param tensor The tensor to add (it's not affected).
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfAdd(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = this;
		for(long i : tensor.getNonZeroElements())
			res.put(i, get(i)+tensor.get(i));
		return res;
	}
	/**
	 * Performs in-memory addition to the Tensor, storing the result in itself.
	 * @param value The value to add to each tensor element.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfAdd(double value) {
		Tensor res = this;
		for(long i=0;i<size();i++)
			res.put(i, get(i)+value);
		return res;
	}
	/**
	 * @param tensor The tensor to subtract
	 * @return a new Tensor that stores the outcome of subtraction
	 */
	public final Tensor subtract(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = copy();
		for(long i : tensor.getNonZeroElements())
			res.put(i, get(i)-tensor.get(i));
		return res;
	}
	/**
	 * Performs in-memory subtraction from the Tensor, storing the result in itself.
	 * @param tensor The tensor to subtract (it's not affected).
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfSubtract(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = this;
		for(long i : tensor.getNonZeroElements()) 
			res.put(i, get(i)-tensor.get(i));
		return res;
	}
	/**
	 * @param tensor The tensor to perform element-wise multiplication with.
	 * @return A new Tensor that stores the outcome of the multiplication.
	 */
	public final Tensor multiply(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = determineZeroCopy(tensor);
		for(long i : getNonZeroElements())
			res.put(i, get(i)*tensor.get(i));
		return res;
	}
	/**
	 * Performs in-memory multiplication on the Tensor, storing the result in itself .
	 * @param tensor The tensor to perform element-wise multiplication with  (it's not affected).
	 * @return <code>this</code> Tensor instance.
	 */
	public Tensor selfMultiply(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = this;
		if(density()<tensor.density()) 
			for(long i : getNonZeroElements())
				res.put(i, get(i)*tensor.get(i));
		else
			for(long i : tensor.getNonZeroElements())
				res.put(i, get(i)*tensor.get(i));
		return res;
	}
	/**
	 * @param value A number to multiply all tensor elements with.
	 * @return A new Tensor that stores the outcome of the multiplication.
	 */
	public final Tensor multiply(double value) {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, get(i)*value);
		return res;
	}
	/**
	 * Performs in-memory multiplication on the Tensor, storing the result to itself.
	 * @param value A number to multiply all tensor elements with.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfMultiply(double value) {
		Tensor res = this;
		for(long i : getNonZeroElements())
			res.put(i, get(i)*value);
		return res;
	}
	/**
	 * Computes the square root of tensor elements.
	 * @return A new Tensor that stores the outcome of finding the absolute square root of each element.
	 */
	public final Tensor sqrt() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, Math.sqrt(Math.abs(get(i))));
		return res;
	}
	/**
	 * Performs in-memory set of each element to the square root of its absolute value.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfSqrt() {
		Tensor res = this;
		for(long i : getNonZeroElements())
			res.put(i, Math.sqrt(Math.abs(get(i))));
		return res;
	}
	/**
	 * Computes the exponential minus 1 of tensor elements.
	 * @return A new Tensor that stores the outcome of finding the operation on each element.
	 */
	public final Tensor expMinusOne() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, Math.exp(get(i)));
		return res;
	}
	/**
	 * Sets the exponential minus 1 of tensor elements.
	 * @return  <code>this</code> Tensor instance.
	 */
	public final Tensor selfExpMinusOne() {
		Tensor res = this;
		for(long i : getNonZeroElements())
			res.put(i, Math.exp(get(i)));
		return res;
	}
	/**
	 * Computes the logarithm of tensor elements.
	 * @return A new Tensor that stores the outcome of finding the logarithm of the absolute of each element.
	 */
	public final Tensor log() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, Math.log(Math.abs(get(i))));
		return res;
	}
	/**
	 * Performs in-memory set of each element to the logarithm of its absolute value.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfLog() {
		Tensor res = this;
		for(long i : getNonZeroElements())
			res.put(i, Math.log(Math.abs(get(i))));
		return res;
	}

	/**
	 * Computes the negative of tensor elements.
	 * @return A new Tensor that stores the outcome of finding the negative of each element.
	 */
	public final Tensor negative() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, -get(i));
		return res;
	}
	/**
	 * Performs in-memory set of each element to the negative of itself.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfNegative() {
		Tensor res = this;
		for(long i : getNonZeroElements())
			res.put(i, -get(i));
		return res;
	}
	/**
	 * Computes the absolute value of tensor elements.
	 * @return A new Tensor that stores the outcome of finding the absolute value of each element.
	 */
	public final Tensor abs() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, Math.abs(get(i)));
		return res;
	}
	/**
	 * Performs in-memory set of each element to its absolute value.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfAbs() {
		Tensor res = this;
		for(long i : getNonZeroElements())
			res.put(i, Math.abs(get(i)));
		return res;
	}
	/**
	 * @return A new Tensor with inversed each non-zero element.
	 */
	public final Tensor inverse() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			if(get(i)!=0)
				res.put(i, 1./get(i));
		return res;
	}
	/**
	 * Performs in-memory the inverse of each non-zero element.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfInverse() {
		Tensor res = this;
		for(long i : getNonZeroElements())
			if(get(i)!=0)
				res.put(i, 1./get(i));
		return res;
	}
	/**
	 * Performs the dot product between this and another tensor.
	 * @param tensor The tensor with which to find the product.
	 * @return The dot product between the tensors.
	 */
	public final double dot(Tensor tensor) {
		assertMatching(tensor);
		double res = 0;
		if(density() < tensor.density())
			for(long i : getNonZeroElements())
				res += get(i)*tensor.get(i);
		else
			for(long i : tensor.getNonZeroElements())
				res += get(i)*tensor.get(i);
		return res;
	}
	/**
	 * Performs the triple dot product between this and two other tensors.
	 * @param tensor1 The firth other tensor with which to find the product.
	 * @param tensor2 The second other tensor with which to find the product.
	 * @return The triple dot product between the tensors.
	 */
	public final double dot(Tensor tensor1, Tensor tensor2) {
		assertMatching(tensor1);
		assertMatching(tensor2);
		double res = 0;
		for(long i : getNonZeroElements())
			res += get(i)*tensor1.get(i)*tensor2.get(i);
		return res;
	}
	/**
	 * @return The L2 norm of the tensor
	 */
	public final double norm() {
		double res = 0;
		for(long i : getNonZeroElements())
			res += get(i)*get(i);
		return Math.sqrt(res);
	}
	/**
	 * @return The sum of tensor elements
	 */
	public final double sum() {
		double res = 0;
		for(long i : getNonZeroElements())
			res += get(i);
		return res;
	}
	/**
	 * Wraps a range of elements within a tensor
	 * without allocating memory anew. Editing the returned
	 * tensor also affects the original one and conversely.
	 * The elements are accessed so that the starting position
	 * is accessed at position 0 of the starting tensor.
	 * @param from The starting position of the subtensor till its end.
	 * @return An {@link AccessSubtensor}.
	 * @see #accessSubtensor(long)
	 */
	public Tensor accessSubtensor(long from) {
		return new AccessSubtensor(this, from);
	}
	/**
	 * Wraps a range of elements within a tensor
	 * without allocating memory anew. Editing the returned
	 * tensor also affects the original one and conversely.
	 * The elements are accessed so that the starting position
	 * is accessed at position 0 of the starting tensor. Accessing
	 * stops up to but not including the end poisition,
	 * so that <code>accessSubtensor(0, size())</code> is
	 * a see-through copy of the original tensor.
	 * @param from The starting position of the subtensor.
	 * @param to The end position of the subtensor that is not included.
	 * @return An {@link AccessSubtensor}.
	 * @see #accessSubtensor(long)
	 */
	public Tensor accessSubtensor(long from, long to) {
		return new AccessSubtensor(this, from, to);
	}
	/**
	 * Computes the maximum tensor element. If the tensor has zero {@link #size()}, 
	 * this returns <code>Double.NEGATIVE_INFINITY</code>.
	 * @return The maximum tensor element
	 * @see #argmax()
	 * @see #min()
	 */
	public final double max() {
		double res = Double.NEGATIVE_INFINITY;
		for(long i=0;i<size;i++) {
			double value = get(i);
			if(value>res)
				res = value;
		}
		return res;
	}
	/**
	 * Computes the position of the maximum tensor element. If the tensor has zero {@link #size()}, 
	 * this returns <code>-1</code>.
	 * @return The position of the maximum tensor element
	 * @see #max()
	 * @see #argmin()
	 */
	public final long argmax() {
		double res = Double.NEGATIVE_INFINITY;
		long pos = -1;
		for(long i=0;i<size;i++) {
			double value = get(i);
			if(value>res) {
				res = value;
				pos = i;
			}
		}
		return pos;
	}
	/**
	 * Computes the minimum tensor element. If the tensor has zero {@link #size()}, 
	 * this returns <code>Double.POSITIVE_INFINITY</code>.
	 * @return The minimum tensor element
	 * @see #argmin()
	 * @see #max()
	 */
	public final double min() {
		double res = Double.POSITIVE_INFINITY;
		for(long i=0;i<size;i++) {
			double value = get(i);
			if(value<res)
				res = value;
		}
		return res;
	}
	/**
	 * Computes the position of the minimum tensor element. If the tensor has zero {@link #size()}, 
	 * this returns <code>-1</code>.
	 * @return The position of the minimum tensor element
	 * @see #min()
	 * @see #argmax()
	 */
	public final long argmin() {
		double res = Double.POSITIVE_INFINITY;
		long pos = -1;
		for(long i=0;i<size;i++) {
			double value = get(i);
			if(value<res) {
				res = value;
				pos = i;
			}
		}
		return pos;
	}
	/**
	 * A string serialization of the tensor that can be used by the constructor {@link DenseTensor#DenseTensor(String)} to create an identical copy.
	 * @return A serialization of the tensor.
	 */
	@Override
	public String toString() {
		StringBuilder res = new StringBuilder();
		if(size()!=0)
			res.append(get(0));
		for(long i=1;i<size();i++)
			res.append(",").append(get(i));
		return res.toString();
	}
	/**
	 * @return A copy of the tensor on which L2 normalization has been performed.
	 * @see #setToNormalized()
	 */
	public final Tensor normalized() {
		double norm = norm();
		Tensor res = zeroCopy();
		if(norm!=0)
			for(long i : getNonZeroElements())
				res.put(i, get(i)/norm);
		return res;
	}
	/**
	 * @return A copy of the tensor on which division with the sum has been performed
	 * (if the tensor contains no negative elements, this is equivalent to L1 normalization)
	 * @see #setToProbability()
	 */
	public final Tensor toProbability() {
		double norm = sum();
		Tensor res = zeroCopy();
		if(norm!=0)
			for(long i : getNonZeroElements())
				res.put(i, get(i)/norm);
		return res;
	}
	/**
	 * L2-normalizes the tensor's elements. Does nothing if the {@link #norm()} is zero.
	 * @return <code>this</code> Tensor instance.
	 * @see #normalized()
	 */
	public final Tensor setToNormalized() {
		double norm = norm();
		if(norm!=0)
			for(long i : getNonZeroElements())
				put(i, get(i)/norm);
		return this;
	}
	/**
	 * Divides the tensor's elements with their sum. Does nothing if the {@link #sum()} is zero.
	 * @return <code>this</code> Tensor instance.
	 * @see #toProbability()
	 */
	public final Tensor setToProbability() {
		double norm = sum();
		if(norm!=0)
			for(long i : getNonZeroElements())
				put(i, get(i)/norm);
		return this;
	}
	/**
	 * Set all tensor element values to 1/{@link #size()}
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToUniform() {
		for(long i=0;i<size();i++)
			put(i, 1./size());
		return this;
	}
	/**
	 * Set all tensor element values to 1.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToOnes() {
		for(long i=0;i<size();i++)
			put(i, 1.);
		return this;
	}
	/**
	 * Set all tensor element values to 0.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToZero() {
		for(long i=0;i<size();i++)
			put(i, 0.);
		return this;
	}
	/**
	 * Retrieves a representation of the Tensor as an array of doubles.
	 * @return An array of doubles
	 */
	public final double[] toArray() {
		double[] values = new double[(int)size()];
		for(long i=0;i<size();i++)
			values[(int)i] = get(i);
		return values;
	}
	/**
	 * Converts a given value to a JGNN-compatible dense tensor.
	 * @param value A given value.
	 * @return a Tensor holding the given value
	 * @see #toDouble()
	 * @see Matrix#fromDouble(double)
	 */
	public static Tensor fromDouble(double value) {
		Tensor ret = new DenseTensor(1);
		ret.put(0, value);
		return ret;
	}
	/**
	 * Creates a dense tensor holding the desired range [start, start+1, ..., end-1].
	 * This allocates a new tensor.
	 * @param start The start of the range.
	 * @param end The end of the range.
	 * @return A {@link DenseTensor} with size end-start
	 * @see #fromRange(long)
	 */
	public static Tensor fromRange(long start, long end) {
		return new DenseTensor(new Range(start, end));
	}
	/**
	 * Creates a dense tensor holding the desired range [0, 1, ..., end-1].
	 * This allocates a new tensor.
	 * @param end The end of the range.
	 * @return A {@link DenseTensor} with size end-start
	 * @see #fromRange(long, long)
	 */
	public static Tensor fromRange(long end) {
		return fromRange(0, end);
	}
	/**
	 * Converts a tensor of {@link #size()}==1 to double. Throws an exception otherwise.
	 * @return A double.
	 * @throws RuntimeException If the tensor is not of size 1.
	 * @see Tensor #fromDouble(double)
	 */
	public double toDouble() {
		assertSize(1);
		return get(0);
	}
	/**
	 * Accesses the tensor through a single-column matrix with the tensor as the only row.
	 * Editing the returned matrix also edits the original tensor.
	 * No new memory is allocated for tensor values.
	 * @return A {@link WrapCols} instance.
	 * @see #asRow()
	 */
	public WrapCols asColumn() {
		return new WrapCols(this);
	}
	/**
	 * Accesses the tensor through a single-row matrix with the tensor as the only column.
	 * Editing the returned matrix also edits the original tensor.
	 * No new memory is allocated for tensor values.
	 * @return A {@link WrapRows} instance.
	 * @see #asColumn()
	 */
	public WrapRows asRow() {
		return new WrapRows(this);
	}
	/**
	 * Describes the type, size and other characteristics of the tensor.
	 * @return A String description.
	 */
	public String describe() {
		return "Tensor ("+(dimensionName==null?"":(dimensionName+" "))+size()+")";
	}
	/**
	 * Performs the equivalent of Java's typecasting that fits
	 * in functional interfaces.
	 * @param <Type> The automatically inferred type of the class.
	 * @param type The class to cast to.
	 * @return <code>this</code> Tensor instance typecast to the given type.
	 */
	@SuppressWarnings("unchecked")
	public <Type> Type cast(Class<Type> type) {
		return (Type)this;
	}
	/**
	 * Automatically determines which between the tensor and a competitor is chosen create zero copies for two-argument operations.
	 * 
	 * @param with The competitor.
	 * @return A zero copy of either the tensor or the competitor.
	 */
	protected Tensor determineZeroCopy(Tensor with) {
		try {
			return zeroCopy(size());
		}
		catch(UnsupportedOperationException e) {
		}
		try {
			return with.zeroCopy(size());
		}
		catch(UnsupportedOperationException e) {
		}
		throw new UnsupportedOperationException("Neither "+describe()+" nor "+with.describe()+" support zeroCopy(rows, cols)");
	}
}
