package mklab.JGNN.core.matrix;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.SparseTensor;

public class SparseSymmetric extends Matrix {
	private Tensor tensor;
	public SparseSymmetric(long rows, long cols) {
		super(rows, cols);
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return new SparseSymmetric(rows, cols);
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
		long rows = getRows();
		return tensor.get(pos) + tensor.get((pos % rows)*rows + pos/rows);
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
		long rows = getRows();
		for(long i : getNonZeroElements())
			ret.add(new AbstractMap.SimpleEntry<Long, Long>(i % rows, i/rows));
		for(long i : getNonZeroElements())
			ret.add(new AbstractMap.SimpleEntry<Long, Long>(i / rows, i%rows));
		return ret;
	}
}