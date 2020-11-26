package mklab.JGNN.core.primitives.matrix;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.primitives.Matrix;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.primitives.tensor.SparseTensor;

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