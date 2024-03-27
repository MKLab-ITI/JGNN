package mklab.JGNN.core.matrix;

import java.util.AbstractMap;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Defines a matrix whose columns are all a copy of a {@link Tensor}.
 * To avoid potential confusion, setting element values (and all supported operations) throws
 * an exception.
 * 
 * @author Emmanouil Krasanakis
 * @see RowRepetition
 */
public class ColumnRepetition extends Matrix {
	protected class Repeat1DIterator implements Iterator<Long>, Iterable<Long> {
		private Iterator<Long> iterator;
		private long current;
		public Repeat1DIterator() {
			this.iterator = column.iterator();
			current = 0;
		}
		@Override
		public boolean hasNext() {
			return current<getRows()-1 || iterator.hasNext();
		}
		@Override
		public Long next() {
			if(!iterator.hasNext()) {
				current += 1;
				iterator = column.iterator();
			}
			long pos = iterator.next();
			return pos*getRows()+current;
		}
		@Override
		public Iterator<Long> iterator() {
			return this;
		}
	}

	protected class Repeat2DIterator implements Iterator<Entry<Long, Long>>, Iterable<Entry<Long, Long>> {
		private Iterator<Long> iterator;
		private long current;
		public Repeat2DIterator() {
			this.iterator = column.iterator();
			current = 0;
		}
		@Override
		public boolean hasNext() {
			return current<getCols()-1 || iterator.hasNext();
		}
		@Override
		public Entry<Long, Long> next() {
			if(!iterator.hasNext()) {
				current += 1;
				iterator = column.iterator();
			}
			long pos = iterator.next();
			return new AbstractMap.SimpleEntry<Long,Long>(Long.valueOf(pos), Long.valueOf(current));
		}
		@Override
		public Iterator<Entry<Long, Long>> iterator() {
			return this;
		}
	}
	
	protected Tensor column;
	/**
	 * Instantiates a matrix repeating a tensor to be treated as a column.
	 * @param times The number of times the column should be repeated.
	 * @param column The column {@link Tensor}.
	 */
	public ColumnRepetition(long times, Tensor column) {
		super(times, column.size());
		this.column = column;
		this.setDimensionName(null, column.getDimensionName());
	}
	/**
	 * Retrieves the wrapped column tensor.
	 * @return The wrapped {@link Tensor}.
	 */
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
		return new Repeat1DIterator();
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Repeat2DIterator();
	}
	@Override
	public void release() {
	}
	@Override
	public void persist() {
		column.persist();
	}
	
}
