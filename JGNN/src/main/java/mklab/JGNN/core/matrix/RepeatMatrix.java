package mklab.JGNN.core.matrix;

import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.RepeatTensor;
import mklab.JGNN.core.util.Range2D;

/**
 * Implements a {@link Matrix} whose elements are all equals.
 * 
 * @author Emmanouil Krasanakis
 */
public class RepeatMatrix extends Matrix {
	private Tensor tensor;

	/**
	 * Generates a dense matrix with the designated number of rows and columns.
	 * 
	 * @param rows The number of rows.
	 * @param cols The number of columns.
	 */
	public RepeatMatrix(double value, long rows, long cols) {
		super(rows, cols);
		tensor = new RepeatTensor(value, rows * cols);
	}

	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return new DenseMatrix(getRows(), getCols());
	}

	@Override
	protected void allocate(long size) {
	}

	@Override
	public Tensor put(long pos, double value) {
		throw new UnsupportedOperationException();
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
		return new Range2D(0, getRows(), 0, getCols());
	}

	@Override
	public void release() {
		tensor.release();
	}

	@Override
	public void persist() {
		tensor.persist();
	}
}