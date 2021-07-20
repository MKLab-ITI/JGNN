package mklab.JGNN.core.matrix;

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
		for(Tensor col : rows)
			col.assertSize(rows.get(0).size());
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Range2D(0, getRows(), 0, getCols());
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
		throw new UnsupportedOperationException();
	}
	@Override
	protected void allocate(long size) {
	}
	@Override
	public Tensor put(long pos, double value) {
		long row = pos % getRows();
		long col = pos / getRows();
		rows.get((int) col).put(row, value);
		return this;
	}
	@Override
	public double get(long pos) {
		long row = pos % getRows();
		long col = pos / getRows();
		return rows.get((int) col).get(row);
	}
	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}
}
