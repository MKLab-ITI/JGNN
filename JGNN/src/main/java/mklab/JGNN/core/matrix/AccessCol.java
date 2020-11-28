package mklab.JGNN.core.matrix;

import java.util.Iterator;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

public class AccessCol extends Tensor {
	private Matrix matrix;
	private long col;
	
	public AccessCol(Matrix matrix, long col) {
		super(matrix.getRows());
		this.matrix = matrix;
	}
	
	@Override
	protected void allocate(long size) {
	}

	@Override
	public Tensor put(long pos, double value) {
		matrix.put(pos, col, value);
		return this;
	}

	@Override
	public double get(long pos) {
		return matrix.get(pos, col);
	}

	@Override
	public Tensor zeroCopy() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}

}
