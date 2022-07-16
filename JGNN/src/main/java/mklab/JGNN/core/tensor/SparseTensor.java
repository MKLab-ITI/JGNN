package mklab.JGNN.core.tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import mklab.JGNN.core.Tensor;

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
	private HashMap<Long, Double> values;
	private List<Long> keySet;
	
	public SparseTensor(long length) {
		super(length);
	}
	public SparseTensor() {
		this(0);
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
			keySet = null;
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
	public Tensor zeroCopy(long size) {
		return new SparseTensor(size);
	}
	@Override
	public synchronized Iterator<Long> traverseNonZeroElements() {
		if(keySet==null)
			keySet = new ArrayList<Long>(values.keySet());
		return keySet.iterator();
	}
}