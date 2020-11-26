package mklab.JGNN.core.primitives;

import java.util.Iterator;

import mklab.JGNN.core.primitives.tensor.DenseTensor;

/**
 * This class provides a native java implementation of Tensor functionalities.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class Tensor implements Iterable<Long> {
	private long size;
	/**
	 * Construct that creates a tensor of zeros given its number of elements
	 * @param size The number of tensor elements
	 */
	public Tensor(long size) {
		init(size);
	}
	protected Tensor() {}
	/**
	 * Set tensor elements to random values in the range [0,1]
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor setToRandom() {
		for(long i=0;i<size();i++)
			put(i, Math.random());
		return this;
	}
	protected final void init(long size) {
		this.size = size;
		allocate(size);
	}
	public void assertFinite() {
		for(long i : getNonZeroElements())
			if(!Double.isFinite(get(i)))
				throw new RuntimeException("Did not find a finite value");
	}
	protected abstract void allocate(long size);
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
	 * @see #put(int, double)
	 */
	public final void putAdd(long pos, double value) {
		put(pos, get(pos)+value);
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
	protected final void assertSize(long size) {
		if(size()!=size)
			throw new RuntimeException("Different sizes: given "+size+" vs "+size());
	}
	/**
	 * Asserts that the tensor's dimensions match with another tensor. This check can be made
	 * more complex by derived classes, but for a base Tensor instance it is equivalent {@link #assertSize(int)}.
	 * This method calls {@link #isMatching(Tensor)} to compare the tensors and throws an exception
	 * if it returns false.
	 * @param other The other tensor to compare with.
	 */
	protected final void assertMatching(Tensor other) {
		if(!isMatching(other)) 
			throw new RuntimeException("Non-compliant: "+describe()+" vs "+other.describe());
	}
	protected boolean isMatching(Tensor other) {
		return size==other.size();
	}
	/**
	 * @return A tensor with the same size but zero elements
	 */
	public abstract Tensor zeroCopy();

	@Override
	public Iterator<Long> iterator() {
		return traverseNonZeroElements();
	}
	
	public final Iterable<Long> getNonZeroElements() {
		return this;
	}
	@SuppressWarnings("unused")
	public final long getNumNonZeroElements() {
		long ret = 0;
		for(long pos : getNonZeroElements())
			ret += 1;
		return ret;
	}
	public final Tensor subtensor(long from, long end) {
		// TODO: implement as "wrapingSubtensor"
		// that can edit the original copy but translates it to the given range
		Tensor ret = new DenseTensor(end-from);
		for(long pos : getNonZeroElements())
			if(pos>=from && pos<end)
				ret.put(pos-from, get(pos));
		return ret;
	}
	/**
	 * @return An iterator that covers the non-zero elements in the elements
	 */
	public abstract Iterator<Long> traverseNonZeroElements();
	/**
	 * Creates a {@link #zeroCopy()} and transfers to it element values.
	 * @return a copy of the Tensor with the same size and contents
	 */
	public final Tensor copy() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, get(i));
		return res;
	}
	/**
	 * @param tensor The tensor to add with
	 * @return a new Tensor that stores the outcome of addition
	 */
	public final Tensor add(Tensor tensor) {
		assertMatching(tensor);
		Tensor res = zeroCopy();
		for(long i : tensor.getNonZeroElements())
			res.put(i, get(i)+tensor.get(i));
		return res;
	}
	/**
	 * @param tensor The value to add to each element
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
	 * @param tensor The value to add to each tensor element.
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
		Tensor res = zeroCopy();
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
		Tensor res = zeroCopy();
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
		for(long i : getNonZeroElements())
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
	 * @return A new Tensor that stores the outcome of finding the absolute square root of each element.
	 */
	public final Tensor sqrt() {
		Tensor res = zeroCopy();
		for(long i : getNonZeroElements())
			res.put(i, Math.sqrt(Math.abs(get(i))));
		return res;
	}
	/**
	 * Performs in-memory the square root of the absolute of each element.
	 * @return <code>this</code> Tensor instance.
	 */
	public final Tensor selfSqrt() {
		Tensor res = this;
		for(long i : getNonZeroElements())
			res.put(i, Math.sqrt(Math.abs(get(i))));
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
		for(long i : getNonZeroElements())
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
	 * A string serialization of the tensor that can be used by the constructor {@link #Tensor(String)} to create an identical copy.
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
	 * L2-normalizes the tensor's elements.
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
	
	public static Tensor fromDouble(double value) {
		Tensor ret = new DenseTensor(1);
		ret.put(0, value);
		return ret;
	}
	
	public double toDouble() {
		assertSize(1);
		return get(0);
	}
	public String describe() {
		return "Tensor ("+size()+")";
	}
}
