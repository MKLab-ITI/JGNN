package mklab.JGNN.core.matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;
import mklab.JGNN.core.util.Range2D;

/**
 * Wraps a list of tensors into a matrix with the tensors as rows.
 * Does not allocate additional elements. Editing the matrix edits
 * the original tensors and conversely.
 * <br>
 * @implSpec TODO for future versions to not use dense matrix iterators. 
 * @author Emmanouil Krasanakis
 */
public class WrapRows extends Matrix {
	private List<Tensor> rows;
	public WrapRows(Tensor... rows) {
		this(Arrays.asList(rows));
	}
	public WrapRows(List<Tensor> rows) {
		super(rows.size(), rows.get(0).size());
		this.rows = rows;
		for(Tensor row : rows)
			row.assertMatching(rows.get(0));
		setRowName(rows.get(0).getDimensionName());
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Range2D(0, getRows(), 0, getCols());
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
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
		return new Range(0, size());
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
