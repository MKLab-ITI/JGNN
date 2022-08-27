package mklab.JGNN.core.matrix;

import java.util.AbstractMap;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Generates a transposed version of a base matrix, with which it shares elements.
 * This avoids excessive memory allocation and can be used to quickly perform
 * operations with a transposed version of a matrix. Prefer using 
 * {@link mklab.JGNN.core.Matrix#asTransposed()}, which wraps usage of this class.
 * 
 * @author Emmanouil Krasanakis
 */
public class TransposedMatrix extends Matrix {
	private Matrix matrix;
	protected class Transposed1DIterator implements Iterator<Long>, Iterable<Long> {
		private Iterator<Long> iterator;
		public Transposed1DIterator(Iterator<Long> iterator) {
			this.iterator = iterator;
		}
		@Override
		public boolean hasNext() {
			return iterator.hasNext();
		}
		@Override
		public Long next() {
			long pos = iterator.next();
			long row = pos % getRows();
			long col = pos / getRows();
			return col+row*getRows(); //transposed of Matrix.put convention
		}
		@Override
		public Iterator<Long> iterator() {
			return this;
		}
	}
	protected class Transposed2DIterator implements Iterator<Entry<Long, Long>>, Iterable<Entry<Long, Long>> {
		private Iterator<Entry<Long, Long>> iterator;
		public Transposed2DIterator(Iterator<Entry<Long, Long>> iterator) {
			this.iterator = iterator;
		}
		@Override
		public boolean hasNext() {
			return iterator.hasNext();
		}
		@Override
		public Entry<Long, Long> next() {
			Entry<Long, Long> origin = iterator.next();
			return new AbstractMap.SimpleEntry<Long,Long>(Long.valueOf(origin.getValue()), Long.valueOf(origin.getKey()));
		}
		@Override
		public Iterator<Entry<Long, Long>> iterator() {
			return this;
		}
	}
	
	public TransposedMatrix(Matrix matrix) {
		super(matrix.getCols(), matrix.getRows());
		this.matrix = matrix;
		setDimensionName(matrix.getDimensionName());
		setRowName(matrix.getColName());
		setColName(matrix.getRowName());
	}

	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Transposed2DIterator(matrix.getNonZeroEntries().iterator());
	}

	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return matrix.zeroCopy(rows, cols);
	}

	@Override
	protected void allocate(long size) {
	}

	@Override
	public Tensor put(long pos, double value) {
		long row = pos % getRows();
		long col = pos / getRows();
		matrix.put(col, row, value);
		return this;
	}

	@Override
	public double get(long pos) {
		long row = pos % getRows();
		long col = pos / getRows();
		return matrix.get(col, row);
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Transposed1DIterator(matrix.traverseNonZeroElements());
	}
	
	@Override
	public Matrix asTransposed() {
		return matrix;
	}

	@Override
	public String describe() {
		return matrix.getClass().getSimpleName()+" ("+getRows()+","+getCols()+")";
	}

	@Override
	public void release() {
	}

	@Override
	public void persist() {
		matrix.persist();
	}
	
}
