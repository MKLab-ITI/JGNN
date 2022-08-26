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
 * Wraps a list of tensors into a matrix with the tensors as columns.
 * Does not allocate additional elements. Editing the matrix edits
 * the original tensors and conversely.
 * <br>
 * @implSpec TODO for future versions to not use dense matrix iterators. 
 * @author Emmanouil Krasanakis
 */
public class WrapCols extends Matrix {
	private List<Tensor> cols;
	public WrapCols(Tensor... cols) {
		this(Arrays.asList(cols));
	}
	public WrapCols(List<Tensor> cols) {
		super(cols.get(0).size(), cols.size());
		this.cols = cols;
		for(Tensor col : cols) 
			col.assertMatching(cols.get(0));
		setColName(cols.get(0).getDimensionName());
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Range2D(0, getRows(), 0, getCols());
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
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
		return new Range(0, size());
	}
	
	@Override
	public Tensor accessCol(long col) {
		return cols.get((int) col);
	}
}
