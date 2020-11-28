package mklab.JGNN.core.tensor;

import java.util.Iterator;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

public class AccessSubtensor extends Tensor {
	private Tensor baseTensor;
	private long begin;
	
	public AccessSubtensor(Tensor baseTensor, long begin) {
		this(baseTensor, begin, baseTensor.size());
	}
	public AccessSubtensor(Tensor baseTensor, long begin, long end) {
		super(end-begin);
		if(baseTensor==null)
			throw new IllegalArgumentException("SubTensor cannot wrap a null base tensor");
		this.baseTensor = baseTensor;
		this.begin = begin;
	}
	@Override
	protected void allocate(long size) {
	}

	@Override
	public Tensor put(long pos, double value) {
		if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		return baseTensor.put(pos+begin, value);
	}

	@Override
	public double get(long pos) {
		if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		return baseTensor.get(pos+begin);
	}

	@Override
	public Tensor zeroCopy() {
		return new AccessSubtensor(baseTensor.zeroCopy(), begin);
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}

}
