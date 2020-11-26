package mklab.JGNN.core.primitives.matrix;

import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.primitives.Matrix;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.primitives.tensor.DenseTensor;
import mklab.JGNN.core.util.Range2D;

public class DenseMatrix extends Matrix {
	private Tensor tensor;
	public DenseMatrix(long rows, long cols) {
		super(rows, cols);
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return new DenseMatrix(rows, cols);
	}
	@Override
	protected void allocate(long size) {
		tensor = new DenseTensor(size);
	}
	@Override
	public Tensor put(long pos, double value) {
		tensor.put(pos, value);
		return this;
	}
	@Override
	public double get(long pos) {
		return tensor.get(pos);
	}
	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return tensor.traverseNonZeroElements();
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Iterable<Entry<Long, Long>>() {
			@Override
			public Iterator<Entry<Long, Long>> iterator() {
				return new Range2D(0, getRows(), 0, getCols());
			}
		};
	}
}