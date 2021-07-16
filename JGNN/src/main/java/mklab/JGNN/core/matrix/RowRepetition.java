package mklab.JGNN.core.matrix;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range2D;

/**
 * Defines a matrix whose rows are all a copy of a {@link Tensor}.
 * To avoid potential confusion, setting element values (and all supported operations) throws
 * an exception.
 * 
 * @author Emmanouil Krasanakis
 * @see ColumnRepetition
 */
public class RowRepetition extends Matrix {
	protected Tensor row;
	public RowRepetition(Tensor row, long times) {
		super(row.size(), times);
		this.row = row;
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return new DenseMatrix(rows, cols);
	}
	@Override
	protected void allocate(long size) {
	}
	@Override
	public Tensor put(long pos, double value) {
		throw new RuntimeException("ColumnRepetion does not support method puts");
	}
	@Override
	public double get(long pos) {
		return row.get(pos % getRows());
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		ArrayList<Long> nonZeros = new ArrayList<Long>();
		for(long col=0;col<getCols();col++)
			for(long rowPos : row.getNonZeroElements()) 
				nonZeros.add(rowPos+col*getRows());
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
