package mklab.JGNN.core.primitives.tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import mklab.JGNN.core.primitives.Tensor;

public class SparseTensor extends Tensor {
	private HashMap<Long, Double> values;
	public SparseTensor(long length) {
		super(length);
	}
	protected SparseTensor() {
	}
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
	public final synchronized double get(long pos) {
		if(pos<0 || pos>=size())
			throw new IllegalArgumentException("Tensor position "+pos+" out of range [0, "+size()+")");
		return values.getOrDefault(pos, 0.);
	}
	@Override
	protected void allocate(long size) {
		values = new HashMap<Long, Double>();
	}
	@Override
	public Tensor zeroCopy() {
		return new SparseTensor(size());
	}
	@Override
	public synchronized Iterator<Long> traverseNonZeroElements() {
		return new ArrayList<Long>(values.keySet()).iterator();
	}
}