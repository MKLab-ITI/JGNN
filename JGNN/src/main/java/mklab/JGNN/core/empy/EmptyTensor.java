package mklab.JGNN.core.empy;

import java.util.ArrayList;
import java.util.Iterator;

import mklab.JGNN.core.Tensor;

public class EmptyTensor extends Tensor {
	public EmptyTensor() {
		super(0);
	}
	public EmptyTensor(long size) {
		super(size);
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
	public Tensor zeroCopy(long size) {
		return this;
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return (new ArrayList<Long>()).iterator();
	}

}
