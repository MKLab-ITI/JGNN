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
 * Wraps a list of tensors into a matrix with the tensors as rows.
 * Does not allocate additional elements. Editing the matrix edits
 * the original tensors and conversely.
 * <br>
 * @author Emmanouil Krasanakis
 */
public class WrapRows extends Matrix {

	protected class Wrap1DIterator implements Iterator<Long>, Iterable<Long> {
		private Iterator<Long> iterator;
		private long current;
		public Wrap1DIterator() {
			this.iterator = rows.get(0).iterator();
			current = 0;
		}
		@Override
		public boolean hasNext() {
			return current<rows.size()-1 || iterator.hasNext();
		}
		@Override
		public Long next() {
			if(!iterator.hasNext()) {
				current += 1;
				iterator = rows.get((int)current).iterator();
			}
			long pos = iterator.next();
			return current+pos*getRows();
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
			this.iterator = rows.get(0).iterator();
			current = 0;
		}
		@Override
		public boolean hasNext() {
			return current<rows.size()-1 || iterator.hasNext();
		}
		@Override
		public Entry<Long, Long> next() {
			if(!iterator.hasNext()) {
				current += 1;
				iterator = rows.get((int)current).iterator();
			}
			long pos = iterator.next();
			return new AbstractMap.SimpleEntry<Long,Long>(Long.valueOf(current), Long.valueOf(pos));
		}
		@Override
		public Iterator<Entry<Long, Long>> iterator() {
			return this;
		}
	}
	
	private List<Tensor> rows;
	private Matrix zeroCopyType;
	private long estimateNonZeroes = 0;
	public WrapRows(Tensor... rows) {
		this(Arrays.asList(rows));
	}
	public WrapRows(List<Tensor> rows) {
		super(rows.size(), rows.get(0).size());
		this.rows = rows;
		estimateNonZeroes = 0;
		for(Tensor row : rows) {
			row.assertMatching(rows.get(0));
			estimateNonZeroes += row.estimateNumNonZeroElements();
		}
		setRowName(rows.get(0).getDimensionName());
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
	public WrapRows setZeroCopyType(Matrix zeroCopyType) {
		this.zeroCopyType = zeroCopyType;
		return this;
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
		if(zeroCopyType!=null)
			return zeroCopyType.zeroCopy(rows, cols);
		if(cols!=getRows() && rows!=getRows())
			throw new UnsupportedOperationException();
		ArrayList<Tensor> newRows = new ArrayList<Tensor>();
		long colSize = rows==getRows()?cols:rows;
		for(Tensor row : this.rows)
			newRows.add(row.zeroCopy(colSize));
		return rows==getRows()?new WrapRows(newRows):new WrapCols(newRows);
	}
	@Override
	protected void allocate(long size) {
	}
	@Override
	public Tensor put(long pos, double value) {
		long row = pos % getRows();
		long col = pos / getRows();
		rows.get((int) row).put(col, value);
		return this;
	}
	@Override
	public double get(long pos) {
		long row = pos % getRows();
		long col = pos / getRows();
		return rows.get((int) row).get(col);
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
	public Tensor accessRow(long row) {
		return rows.get((int) row);
	}
	@Override
	public void release() {
	}
	@Override
	public void persist() {
		for(Tensor row : rows)
			row.persist();
	}
}
