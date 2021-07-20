package mklab.JGNN.core.matrix;

import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Range2D;

/**
 * Implements a dense {@link Matrix} where all elements are stored in memory.
 * For matrices with more than MAXINT number of elements or many zeros use the {@link SparseMatrix}
 * structure.
 * 
 * @author Emmanouil Krasanakis
 */
public class DenseMatrix extends Matrix {
	private Tensor tensor;
	/**
	 * Generates a dense matrix with the designated number of rows and columns.
	 * @param rows The number of rows.
	 * @param cols The number of columns.
	 */
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
		return new Range2D(0, getRows(), 0, getCols());
	}
}