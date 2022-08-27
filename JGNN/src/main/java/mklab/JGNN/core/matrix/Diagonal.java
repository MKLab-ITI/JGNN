package mklab.JGNN.core.matrix;

import java.util.AbstractMap;
import java.util.Iterator;
import java.util.Map.Entry;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Implements a square matrix whose diagonal elements are determined by the correspond values of
 * an underlying tensor and off-diagonal elements are zero. Elements are shared between the matrix
 * and its diagonal tensor. This structure is similar to a sparse matrix.
 * @author Emmanouil Krasanakis
 */
public class Diagonal extends Matrix {
	private Tensor diagonal;

	protected class Diagonal1DIterator implements Iterator<Long>, Iterable<Long> {
		private Iterator<Long> iterator;
		public Diagonal1DIterator(Iterator<Long> iterator) {
			this.iterator = iterator;
		}
		@Override
		public boolean hasNext() {
			return iterator.hasNext();
		}
		@Override
		public Long next() {
			long pos = iterator.next();
			return pos+pos*getRows();
		}
		@Override
		public Iterator<Long> iterator() {
			return this;
		}
	}

	protected class Diagonal2DIterator implements Iterator<Entry<Long, Long>>, Iterable<Entry<Long, Long>> {
		private Iterator<Long> iterator;
		public Diagonal2DIterator(Iterator<Long> iterator) {
			this.iterator = iterator;
		}
		@Override
		public boolean hasNext() {
			return iterator.hasNext();
		}
		@Override
		public Entry<Long, Long> next() {
			long pos = iterator.next();
			return new AbstractMap.SimpleEntry<Long,Long>(Long.valueOf(pos), Long.valueOf(pos));
		}
		@Override
		public Iterator<Entry<Long, Long>> iterator() {
			return this;
		}
	}
	
	protected Diagonal(Tensor diagonal) {
		super(diagonal.size(), diagonal.size());
		this.diagonal = diagonal;
	}

	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Diagonal2DIterator(diagonal.iterator());
	}

	@Override
	public Matrix zeroCopy(long rows, long cols) {
		if(rows!=cols)
			throw new UnsupportedOperationException("Zero copies of diagonal matrices should be square matrices");
		return new Diagonal(diagonal.zeroCopy(rows));
	}

	@Override
	protected void allocate(long size) {
	}

	@Override
	public Tensor put(long pos, double value) {
		long row = pos % getRows();
		long col = pos / getRows();
		if(row!=col)
			throw new UnsupportedOperationException("Cannot put values in off-diagonal elements of diagonal matrices");
		diagonal.put(row, value);
		return this;
	}

	@Override
	public double get(long pos) {
		long row = pos % getRows();
		long col = pos / getRows();
		if(row==col)
			return diagonal.get(row);
		return 0;
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Diagonal1DIterator(diagonal.iterator());
	}

	@Override
	public void release() {
	}

	@Override
	public void persist() {
	}
}
