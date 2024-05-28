package mklab.JGNN.core.tensor;

import java.util.Iterator;

import mklab.JGNN.core.Tensor;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;

/**
 * This class provides a sparse {@link Tensor} with many zero elements.
 * Critically, it implements a {@link #traverseNonZeroElements()} method
 * that provides the positions of only non-zero elements to speed up computations.
 * 
 * Speed ups are expected mostly for operations between sparse tensors,
 * when sparse tensors are added or subtracted TO tense ones and when
 * sparse tensors are multiplied WITH dense ones.
 * 
 * @author Emmanouil Krasanakis
 */
public class SparseTensor extends Tensor {
	private Long2DoubleOpenHashMap values;
	
	public SparseTensor(long length) {
		super(length);
	}
	public SparseTensor() {
		this(0);
	}
	@Override
	public final synchronized Tensor put(long pos, double value) {
		if(!Double.isFinite(value))
			throw new IllegalArgumentException("Cannot accept non-finite (NaN or Infinity) tensor values");
		else if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		else {
			if(value==0)
				values.remove(pos);
			else
				values.put(pos, value);
		}
		return this;
	}
	@Override
	public final synchronized double get(long pos) {
		if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		return values.get(pos);
	}
	@Override
	protected void allocate(long size) {
		values = new Long2DoubleOpenHashMap((int)Math.min(Math.sqrt(size), Integer.MAX_VALUE), 0.75f);
	}
	@Override
	public Tensor zeroCopy(long size) {
		return new SparseTensor(size);
	}
	@Override
	public synchronized Iterator<Long> traverseNonZeroElements() {
		return values.keySet().iterator();
	}
	@Override
	public long estimateNumNonZeroElements() {
		return values.size();
	}
	@Override
	public void release() {
		values = null;
	}
	@Override
	public void persist() {
	}
}