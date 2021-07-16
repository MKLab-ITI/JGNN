package mklab.JGNN.core.matrix;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range2D;

/**
 * Defines a matrix whose columns are all a copy of a {@link Tensor}.
 * To avoid potential confusion, setting element values (and all supported operations) throws
 * an exception.
 * 
 * @author Emmanouil Krasanakis
 * @see RowRepetition
 */
public class ColumnRepetition extends Matrix {
	protected Tensor column;
	public ColumnRepetition(long times, Tensor column) {
		super(times, column.size());
		this.column = column;
	}
	public Tensor getColumn() {
		return column;
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
		throw new RuntimeException("ColumnRepetion does not support changing base column values. Consider using getColumn().put(...)");
	}
	@Override
	public double get(long pos) {
		return column.get(pos/getRows());
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		ArrayList<Long> nonZeros = new ArrayList<Long>();
		for(long row=0;row<getRows();row++)
			for(long col : column.getNonZeroElements())
				nonZeros.add(row+col*getRows());
		return nonZeros.iterator();
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
