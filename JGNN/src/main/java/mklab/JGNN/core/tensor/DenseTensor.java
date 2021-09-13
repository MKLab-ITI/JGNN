package mklab.JGNN.core.tensor;

import java.util.Iterator;
import java.util.List;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * This class provides a dense {@link Tensor} that wraps an array of doubles.
 * 
 * @author Emmanouil Krasanakis
 */
public class DenseTensor extends Tensor {
	private double[] values;

	public DenseTensor(List<? extends Number> values) {
		this(values.size());
		for(int i=0;i<values.size();i++)
			put(i, (double)values.get(i));
	}
	public DenseTensor(double... values) {
		this(values.length);
		for(int i=0;i<values.length;i++)
			put(i, values[i]);
	}
	public DenseTensor(long length) {
		super(length);
	}
	/**
	 * Reconstructs a serialized Tensor (i.e. the outcome of {@link #toString()})
	 * @param expr A serialized tensor
	 * @throws IllegalArgumentException If the serialization is null or empty.
	 */
	public DenseTensor(String expr) {
		if(expr==null)
			throw new IllegalArgumentException("Cannot create tensor from a null string");
		if(expr.length()==0) {
			init(0);
			return;
		}
		String[] splt = expr.split(",");
		init(splt.length);
		for(int i=0;i<splt.length;i++)
			put(i, Double.parseDouble(splt[i]));
	}
	public DenseTensor() {
		this(0);
	}
	public final synchronized Tensor put(long pos, double value) {
		if(!Double.isFinite(value))
			throw new IllegalArgumentException("Cannot accept non-finite (NaN or Infinity) tensor values");
		else if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		else
			values[(int)pos] = value;
		return this;
	}
	public final synchronized double get(long pos) {
		if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		return values[(int)pos];
	}
	@Override
	protected void allocate(long size) {
		values = new double[(int)size];
	}
	@Override
	public Tensor zeroCopy(long size) {
		return new DenseTensor(size);
	}
	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}
}