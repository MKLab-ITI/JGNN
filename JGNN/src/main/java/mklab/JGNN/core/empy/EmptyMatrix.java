package mklab.JGNN.core.empy;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

public class EmptyMatrix extends Matrix {
	public EmptyMatrix(long rows, long cols) {
		super(rows, cols);
	}

	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return (new ArrayList<Entry<Long, Long>>());
	}

	@Override
	public Matrix zeroCopy(long rows, long cols) {
		return new EmptyMatrix(rows, cols);
	}

	@Override
	protected void allocate(long size) {
	}

	@Override
	public void release() {
	}

	@Override
	public void persist() {
	}

	@Override
	public Tensor put(long pos, double value) {
		return this;
	}

	@Override
	public double get(long pos) {
		return 0;
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return (new ArrayList<Long>()).iterator();
	}

}
