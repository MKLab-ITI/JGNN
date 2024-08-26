package mklab.JGNN.core.empty;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * A {@link Matrix} without data that contains only the correct dimension names
 * and sizes. All its data are considered zero. Empty data types try to
 * pervasively fill all operation outcomes in which it is involved. The intent
 * is to use them during
 * {@link mklab.JGNN.adhoc.ModelBuilder#autosize(java.util.List)} to make it
 * lightweight.
 * 
 * @author Emmanouil Krasanakis
 * @see EmptyTensor
 */
public class EmptyMatrix extends Matrix {
	/**
	 * Initializes an {@link EmptyMatrix} of given dimensions. It does not allocate
	 * memory for data.
	 * 
	 * @param rows The number of matrix rows.
	 * @param cols The number of matrix columns.
	 */
	public EmptyMatrix(long rows, long cols) {
		super(rows, cols);
	}

	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return (new ArrayList<Entry<Long, Long>>());
	}

	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return new EmptyMatrix(rows, cols);
	}

	@Override
	protected void allocate(long size) {
	}

	@Override
	public void release() {
	}

	@Override
	public void persist() {
	}

	@Override
	public Tensor put(long pos, double value) {
		return this;
	}

	@Override
	public double get(long pos) {
		return 0;
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return (new ArrayList<Long>()).iterator();
	}

}
