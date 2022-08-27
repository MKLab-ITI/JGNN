package mklab.JGNN.core.tensor;

import java.util.Iterator;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * This class provides {@link Tensor} whose elements are all equal.
 * Due to uncertain usage, its {@link #put(long, double)} operation is unsupported and throws a corresponding exception.
 * Otherwise, instances of this class behave similarly to {@link DenseTensor} but permanently allocate only O(1) memory.
 * 
 * @author Emmanouil Krasanakis
 */
public class RepeatTensor extends Tensor {
	private double value;
	public RepeatTensor(double value, long length) {
		super(length);
		this.value = value;
		if(!Double.isFinite(value))
			throw new IllegalArgumentException("Cannot accept non-finite (NaN or Infinity) tensor values");
	}
	public final synchronized Tensor put(long pos, double value) {
		throw new UnsupportedOperationException("Cannot edit a RepeatTensor: create a new one");
	}
	public final synchronized double get(long pos) {
		if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		return value;
	}
	@Override
	protected void allocate(long size) {
	}
	@Override
	public Tensor zeroCopy(long size) {
		throw new UnsupportedOperationException("Can not copy a RepeatTensor in any way: create a new one");
	}
	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}
	@Override
	public void release() {
	}
	@Override
	public void persist() {
	}
}