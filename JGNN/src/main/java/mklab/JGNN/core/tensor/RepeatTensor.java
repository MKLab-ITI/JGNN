package mklab.JGNN.core.tensor;

import java.util.Iterator;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

public class RepeatTensor extends Tensor {
	private double value;
	public RepeatTensor(double value, long length) {
		super(length);
		this.value = value;
	}
	protected RepeatTensor() {
	}
	public final synchronized Tensor put(long pos, double value) {
		throw new UnsupportedOperationException();
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
	public Tensor zeroCopy() {
		return new RepeatTensor(value, size());
	}
	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}
}