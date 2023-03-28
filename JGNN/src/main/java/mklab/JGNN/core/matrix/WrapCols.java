package mklab.JGNN.core.matrix;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Wraps a list of tensors into a matrix with the tensors as columns.
 * Does not allocate additional elements. Editing the matrix edits
 * the original tensors and conversely.
 * <br>
 * @author Emmanouil Krasanakis
 */
public class WrapCols extends Matrix {

	protected class Wrap1DIterator implements Iterator<Long>, Iterable<Long> {
		private Iterator<Long> iterator;
		private long current;
		public Wrap1DIterator() {
			this.iterator = cols.get(0).iterator();
			current = 0;
		}
		@Override
		public boolean hasNext() {
			while(!iterator.hasNext() && current<cols.size()-1) {
				current += 1;
				iterator = cols.get((int)current).iterator();
			}
			return current<cols.size()-1 || iterator.hasNext();
		}
		@Override
		public Long next() {
			long pos = iterator.next();
			return pos+current*getRows();
		}
		@Override
		public Iterator<Long> iterator() {
			return this;
		}
	}

	protected class Wrap2DIterator implements Iterator<Entry<Long, Long>>, Iterable<Entry<Long, Long>> {
		private Iterator<Long> iterator;
		private long current;
		public Wrap2DIterator() {
			this.iterator = cols.get(0).iterator();
			current = 0;
		}
		@Override
		public boolean hasNext() {
			while(!iterator.hasNext() && current<cols.size()-1) {
				current += 1;
				iterator = cols.get((int)current).iterator();
			}
			return current<cols.size()-1 || iterator.hasNext();
		}
		@Override
		public Entry<Long, Long> next() {
			long pos = iterator.next();
			return new AbstractMap.SimpleEntry<Long,Long>(Long.valueOf(pos), Long.valueOf(current));
		}
		@Override
		public Iterator<Entry<Long, Long>> iterator() {
			return this;
		}
	}
	
	private List<Tensor> cols;
	private Matrix zeroCopyType;
	private long estimateNonZeroes;
	public WrapCols(Tensor... cols) {
		this(Arrays.asList(cols));
	}
	public WrapCols(List<Tensor> cols) {
		super(cols.get(0).size(), cols.size());
		this.cols = cols;
		estimateNonZeroes = 0;
		for(Tensor col : cols) { 
			col.assertMatching(cols.get(0));
			estimateNonZeroes += col.estimateNumNonZeroElements();
		}
		setColName(cols.get(0).getDimensionName());
	}
	@Override
	public long estimateNumNonZeroElements() {
		return estimateNonZeroes;
	}
	/**
	 * Sets a prototype matrix from which to borrow copying operations.
	 * @param zeroCopyType A {@link Matrix} instance from which to borrow {@link #zeroCopy(long, long)}.
	 * @return <code>this</code> object
	 */
	public WrapCols setZeroCopyType(Matrix zeroCopyType) {
		this.zeroCopyType = zeroCopyType;
		return this;
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
		if(zeroCopyType!=null)
			return zeroCopyType.zeroCopy(rows, cols);
		if(cols!=getCols() && rows!=getCols())
			throw new UnsupportedOperationException();
		long rowSize = cols==getCols()?rows:cols;
		ArrayList<Tensor> newCols = new ArrayList<Tensor>();
		for(Tensor col : this.cols)
			newCols.add(col.zeroCopy(rowSize));
		return cols==getCols()?new WrapCols(newCols):new WrapRows(newCols);
	}
	@Override
	protected void allocate(long size) {
	}
	@Override
	public Tensor put(long pos, double value) {
		long row = pos % getRows();
		long col = pos / getRows();
		cols.get((int) col).put(row, value);
		return this;
	}
	@Override
	public double get(long pos) {
		long row = pos % getRows();
		long col = pos / getRows();
		return cols.get((int) col).get(row);
	}
	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Wrap1DIterator();
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Wrap2DIterator();
	}
	
	@Override
	public Tensor accessCol(long col) {
		return cols.get((int) col);
	}
	@Override
	public void release() {
	}
	@Override
	public void persist() {
		for(Tensor col : cols)
			col.persist();
	}
}
