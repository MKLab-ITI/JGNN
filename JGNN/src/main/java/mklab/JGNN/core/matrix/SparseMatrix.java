package mklab.JGNN.core.matrix;

import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.SparseTensor;
import mklab.JGNN.core.util.FastEntry;

/**
 * A sparse {@link Matrix} that allocates memory only for non-zero elements.
 * Operations that involve all matrix elements are slower compared to a
 * {@link DenseMatrix}.
 * 
 * @author Emmanouil Krasanakis
 */
public class SparseMatrix extends Matrix {
	private SparseTensor tensor;

	/**
	 * Generates a sparse matrix with the designated number of rows and columns.
	 * 
	 * @param rows The number of rows.
	 * @param cols The number of columns.
	 */
	public SparseMatrix(long rows, long cols) {
		super(rows, cols);
	}

	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return new SparseMatrix(rows, cols);
	}

	@Override
	protected void allocate(long size) {
		tensor = new SparseTensor(size);
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
	public String describe() {
		return super.describe() + " " + estimateNumNonZeroElements() + "/" + (getRows() * getCols()) + " entries";
	}

	@Override
	public long estimateNumNonZeroElements() {
		return tensor.estimateNumNonZeroElements();
	}

	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		/*
		 * ArrayList<Entry<Long, Long>> ret = new ArrayList<Entry<Long, Long>>();
		 * for(long i : getNonZeroElements()) ret.add(new AbstractMap.SimpleEntry<Long,
		 * Long>(i % getRows(), i/getRows())); return ret;
		 */
		return new Sparse2DIterator(traverseNonZeroElements());
	}

	@Override
	public void release() {
		tensor.release();
	}

	@Override
	public void persist() {
		tensor.persist();
	}

	protected class Sparse2DIterator implements Iterator<Entry<Long, Long>>, Iterable<Entry<Long, Long>> {
		private Iterator<Long> iterator;
		private long rows;
		private final FastEntry<Long, Long> ret = new FastEntry<Long, Long>();

		public Sparse2DIterator(Iterator<Long> iterator) {
			this.iterator = iterator;
			rows = getRows();
		}

		@Override
		public boolean hasNext() {
			return iterator.hasNext();
		}

		@Override
		public Entry<Long, Long> next() {
			long pos = iterator.next();
			ret.setKey(pos % rows);
			ret.setValue(pos / rows);
			return ret;
			// return new AbstractMap.SimpleEntry<Long,Long>(pos % rows, pos/rows);
		}

		@Override
		public Iterator<Entry<Long, Long>> iterator() {
			return this;
		}
	}
}