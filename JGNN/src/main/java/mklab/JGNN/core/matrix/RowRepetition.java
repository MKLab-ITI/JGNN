package mklab.JGNN.core.matrix;

import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.FastEntry;

/**
 * Defines a matrix whose rows are all a copy of a {@link Tensor}. To avoid
 * potential confusion, setting element values (and all supported operations)
 * throws an exception.
 * 
 * @author Emmanouil Krasanakis
 * @see ColumnRepetition
 */
public class RowRepetition extends Matrix {
	protected class Repeat1DIterator implements Iterator<Long>, Iterable<Long> {
		private Iterator<Long> iterator;
		private long current;

		public Repeat1DIterator() {
			this.iterator = row.iterator();
			current = 0;
		}

		@Override
		public boolean hasNext() {
			return current < getCols() - 1 || iterator.hasNext();
		}

		@Override
		public Long next() {
			if (!iterator.hasNext()) {
				current += 1;
				iterator = row.iterator();
			}
			long pos = iterator.next();
			return current + pos * getRows();
		}

		@Override
		public Iterator<Long> iterator() {
			return this;
		}
	}

	protected class Repeat2DIterator implements Iterator<Entry<Long, Long>>, Iterable<Entry<Long, Long>> {
		private Iterator<Long> iterator;
		private long current;
		private final FastEntry<Long, Long> ret = new FastEntry<Long, Long>();

		public Repeat2DIterator() {
			this.iterator = row.iterator();
			current = 0;
		}

		@Override
		public boolean hasNext() {
			return current < getCols() - 1 || iterator.hasNext();
		}

		@Override
		public Entry<Long, Long> next() {
			if (!iterator.hasNext()) {
				current += 1;
				iterator = row.iterator();
			}
			long pos = iterator.next();
			ret.setKey(current);
			ret.setValue(pos);
			return ret;
			// return new AbstractMap.SimpleEntry<Long,Long>(Long.valueOf(current),
			// Long.valueOf(pos));
		}

		@Override
		public Iterator<Entry<Long, Long>> iterator() {
			return this;
		}
	}

	protected Tensor row;

	/**
	 * Instantiates a matrix repeating a tensor to be treated as a row.
	 * 
	 * @param column The row {@link Tensor}.
	 * @param times  The number of times the row should be repeated.
	 */
	public RowRepetition(Tensor row, long times) {
		super(row.size(), times);
		this.row = row;
		this.setDimensionName(row.getDimensionName(), null);
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
		row.persist();
	}

}
