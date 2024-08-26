package mklab.JGNN.core.tensor;

import java.util.ArrayList;
import java.util.Iterator;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * This class provides a dense {@link Tensor} that wraps an array of doubles.
 * 
 * @author Emmanouil Krasanakis
 */
public class DenseTensor extends Tensor {
	public double[] values;

	/**
	 * Constructs a dense tensor from an iterator holding that outputs its values.
	 * Tensor size is equal to the number of extracted values.
	 * 
	 * @param iterator The iterator to obtain values from.
	 */
	public DenseTensor(Iterator<? extends Number> iterator) {
		ArrayList<Number> list = new ArrayList<Number>();
		iterator.forEachRemaining(list::add);
		init(list.size());
		for (int i = 0; i < list.size(); ++i)
			values[i] = list.get(i).doubleValue();
	}

	public DenseTensor(double... values) {
		this(values.length);
		System.arraycopy(values, 0, this.values, 0, values.length);
	}

	/**
	 * Constructs a dense tensor holding zero values.
	 * 
	 * @param size The size of the tensor.
	 */
	public DenseTensor(long size) {
		super(size);
	}

	/**
	 * Reconstructs a serialized Tensor (i.e. the outcome of {@link #toString()})
	 * 
	 * @param expr A serialized tensor
	 * @throws IllegalArgumentException If the serialization is null or empty.
	 */
	public DenseTensor(String expr) {
		if (expr == null)
			throw new IllegalArgumentException("Cannot create tensor from a null string");
		if (expr.length() == 0) {
			init(0);
			return;
		}
		String[] splt = expr.split(",");
		init(splt.length);
		for (int i = 0; i < splt.length; ++i)
			values[i] = Double.parseDouble(splt[i]);
	}

	public DenseTensor() {
		this(0);
	}

	public final Tensor put(long pos, double value) {
		values[(int) pos] = value;
		return this;
	}

	/**
	 * Overloads {@link #put(long, double)} to accept integer positions. Using this
	 * method lets JVM speed up some code.
	 * 
	 * @param pos   The position of the tensor element.
	 * @param value The value to assign.
	 * @return <code>this</code> Tensor instance.
	 * @see #put(long, double)
	 */
	public final Tensor put(int pos, double value) {
		values[pos] = value;
		return this;
	}

	/**
	 * Overloads {@link #putAdd(long, double)} to accept integer positions. Using
	 * this method lets JVM speed up some code.
	 * 
	 * @param pos   The position of the tensor element.
	 * @param value The value to add.
	 * @return <code>this</code> Tensor instance.
	 * @see #put(long, double)
	 */
	public final Tensor putAdd(int pos, double value) {
		values[pos] += value;
		return this;
	}

	public final double get(long pos) {
		return values[(int) pos];
	}

	public final double get(int pos) {
		return values[pos];
	}

	@Override
	protected void allocate(long size) {
		values = new double[(int) size];
	}

	@Override
	public Tensor zeroCopy(long size) {
		if (size >= 100000 && vectorization)
			return new VectorizedTensor(size);
		return new DenseTensor(size);
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}

	@Override
	public void release() {
		values = null;
	}

	@Override
	public void persist() {
	}

	@Override
	public Tensor add(Tensor tensor) {
		assertMatching(tensor);
		if (tensor instanceof DenseTensor) {
			DenseTensor other = (DenseTensor) tensor;
			DenseTensor res = new DenseTensor(size());
			for (int i = 0; i < values.length; ++i)
				res.values[i] = values[i] + other.values[i];
			return res;
		}
		return super.add(tensor);
	}

	@Override
	public Tensor add(double value) {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i)
			res.values[i] = values[i] + value;
		return res;
	}

	@Override
	public Tensor selfAdd(Tensor tensor) {
		assertMatching(tensor);
		if (tensor instanceof DenseTensor) {
			DenseTensor other = (DenseTensor) tensor;
			for (int i = 0; i < values.length; ++i)
				values[i] += other.values[i];
			return this;
		}
		return super.selfAdd(tensor);
	}

	@Override
	public Tensor selfAdd(double value) {
		for (int i = 0; i < values.length; ++i)
			values[i] += value;
		return this;
	}

	@Override
	public Tensor subtract(Tensor tensor) {
		assertMatching(tensor);
		if (tensor instanceof DenseTensor) {
			DenseTensor other = (DenseTensor) tensor;
			DenseTensor res = new DenseTensor(size());
			for (int i = 0; i < values.length; ++i)
				res.values[i] = values[i] - other.values[i];
			return res;
		}
		return super.subtract(tensor);
	}

	@Override
	public Tensor selfSubtract(Tensor tensor) {
		assertMatching(tensor);
		if (tensor instanceof DenseTensor) {
			DenseTensor other = (DenseTensor) tensor;
			for (int i = 0; i < values.length; ++i)
				values[i] -= other.values[i];
			return this;
		}
		return super.selfSubtract(tensor);
	}

	@Override
	public Tensor multiply(Tensor tensor) {
		assertMatching(tensor);
		if (tensor instanceof DenseTensor) {
			DenseTensor other = (DenseTensor) tensor;
			DenseTensor res = new DenseTensor(size());
			for (int i = 0; i < values.length; ++i)
				res.values[i] = values[i] * other.values[i];
			return res;
		}
		return super.multiply(tensor);
	}

	@Override
	public Tensor multiply(double value) {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i)
			res.values[i] = values[i] * value;
		return res;
	}

	@Override
	public Tensor selfMultiply(Tensor tensor) {
		assertMatching(tensor);
		if (tensor instanceof DenseTensor) {
			DenseTensor other = (DenseTensor) tensor;
			for (int i = 0; i < values.length; ++i)
				values[i] *= other.values[i];
			return this;
		}
		return super.selfMultiply(tensor);
	}

	@Override
	public Tensor selfMultiply(double value) {
		for (int i = 0; i < values.length; ++i)
			values[i] *= value;
		return this;
	}

	@Override
	public Tensor sqrt() {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i)
			res.values[i] = Math.sqrt(Math.abs(values[i]));
		return res;
	}

	@Override
	public Tensor selfSqrt() {
		for (int i = 0; i < values.length; ++i)
			values[i] = Math.sqrt(Math.abs(values[i]));
		return this;
	}

	@Override
	public Tensor expMinusOne() {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i)
			res.values[i] = Math.exp(values[i]);
		return res;
	}

	@Override
	public Tensor selfExpMinusOne() {
		for (int i = 0; i < values.length; ++i)
			values[i] = Math.exp(values[i]);
		return this;
	}

	@Override
	public Tensor log() {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i)
			res.values[i] = Math.log(Math.abs(values[i]));
		return res;
	}

	@Override
	public Tensor selfLog() {
		for (int i = 0; i < values.length; ++i)
			values[i] = Math.log(Math.abs(values[i]));
		return this;
	}

	@Override
	public Tensor negative() {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i)
			res.values[i] = -values[i];
		return res;
	}

	@Override
	public Tensor selfNegative() {
		for (int i = 0; i < values.length; ++i)
			values[i] = -values[i];
		return this;
	}

	@Override
	public Tensor abs() {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i)
			res.values[i] = Math.abs(values[i]);
		return res;
	}

	@Override
	public Tensor selfAbs() {
		for (int i = 0; i < values.length; ++i)
			values[i] = Math.abs(values[i]);
		return this;
	}

	@Override
	public Tensor inverse() {
		DenseTensor res = new DenseTensor(size());
		for (int i = 0; i < values.length; ++i) {
			if (values[i] != 0)
				res.values[i] = 1. / values[i];
		}
		return res;
	}

	@Override
	public Tensor selfInverse() {
		for (int i = 0; i < values.length; ++i) {
			if (values[i] != 0)
				values[i] = 1. / values[i];
		}
		return this;
	}
}
