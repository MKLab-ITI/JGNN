package mklab.JGNN.core.matrix;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.SparseTensor;

/**
 * A sparse {@link Matrix} that allocates memory only for non-zero elements. Operations
 * that involve all matrix elements are slower compared to a {@link DenseMatrix}.
 * 
 * @author Emmanouil Krasanakis
 */
public class SparseMatrix extends Matrix {
	private Tensor tensor;
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
		return super.describe()+" "+getNumNonZeroElements()+"/"+(getRows()*getCols())+" entries";
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		ArrayList<Entry<Long, Long>> ret = new ArrayList<Entry<Long, Long>>();
		for(long i : getNonZeroElements())
			ret.add(new AbstractMap.SimpleEntry<Long, Long>(i % getRows(), i/getRows()));
		return ret;
	}
}