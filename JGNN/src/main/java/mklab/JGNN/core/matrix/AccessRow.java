package mklab.JGNN.core.matrix;

import java.util.Iterator;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * Accesses a row of a {@link Matrix} as if it were a dense {@link Tensor}.
 * Prefer using {@link mklab.JGNN.core.Matrix#accessRow(long)}, which wraps usage
 * of this class. Instances of this class share elements with the matrix which
 * they access and do <i>not</i> allocate new memory.
 * 
 * @author Emmanouil Krasanakis
 * @see AccessCol
 */
public class AccessRow extends Tensor {
	private Matrix matrix;
	private long row;

	/**
	 * Instantiates a see-through access of a matrix row.
	 * @param matrix The base matrix.
	 * @param row Which row to access.
	 */
	public AccessRow(Matrix matrix, long row) {
		super(matrix.getCols());
		this.matrix = matrix;
		this.row = row;
		this.setDimensionName(matrix.getRowName());
		if(row<0 || row>=matrix.getRows())
			throw new IllegalArgumentException("Row "+row+" does not exist in "+matrix.describe());
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
	public Tensor zeroCopy(long size) {
		throw new UnsupportedOperationException();
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
