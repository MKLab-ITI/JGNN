package mklab.JGNN.core.tensor;

import java.util.Iterator;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * Wraps a base {@link Tensor} by traversing only its elements in a specified range (from begin, up to end-1).
 * Although in principle it does not require a specific type of base tensor, it is created with optimized
 * {@link DenseTensor} operations in mind. That is, it implements {@link #traverseNonZeroElements()} as a {@link Range}.
 * This class's {@link #zeroCopy()} is marked as unimplemented by throwing an exception, which will also make dependent
 * operations fail. However, it makes sense that members of this class are only used to access (or modify) the subtensor.
 * 
 * @author Emmanouil Krasanakis
 */

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
		if(begin>end)
			throw new IllegalArgumentException("SubTensor cannot start after its end");
		if(begin<0)
			throw new IllegalArgumentException("SubTensor cannot start before zero");
		if(end>baseTensor.size())
			throw new IllegalArgumentException("SubTensor cannot have an end position "+end+" after base tensor size "+baseTensor.size());
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
		baseTensor.put(pos+begin, value);
		return this;
	}

	@Override
	public double get(long pos) {
		if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		return baseTensor.get(pos+begin);
	}

	@Override
	public Tensor zeroCopy(long size) {
		return baseTensor.zeroCopy(size);
	}

	@Override
	public Iterator<Long> traverseNonZeroElements() {
		return new Range(0, size());
	}

}
