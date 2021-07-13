package mklab.JGNN.core.matrix;

import java.util.Iterator;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * Can be used to access a row of a {@link Matrix} as if it were a dense {@link Tensor}.
 * 
 * @author Emmanouil Krasanakis
 */
public class AccessRow extends Tensor {
	private Matrix matrix;
	private long row;
	
	public AccessRow(Matrix matrix, long row) {
		super(matrix.getCols());
		this.matrix = matrix;
	}
	
	@Override
	protected void allocate(long size) {
	}

	@Override
	public Tensor put(long pos, double value) {
		matrix.put(row, pos, value);
		return this;
	}

	@Override
	public double get(long pos) {
		return matrix.get(row, pos);
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
