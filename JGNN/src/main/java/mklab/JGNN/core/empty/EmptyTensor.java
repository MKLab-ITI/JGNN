package mklab.JGNN.core.empty;

import java.util.ArrayList;
import java.util.Iterator;

import mklab.JGNN.core.Tensor;

/**
 * A {@link Tensor} without data that contains only the correct dimension names
 * and sizes. All its data are considered zero. Empty data types try to
 * pervasively fill all operation outcomes in which it is involved. The intent
 * is to use them during
 * {@link mklab.JGNN.adhoc.ModelBuilder#autosize(java.util.List)} to make it
 * lightweight.
 * 
 * @author Emmanouil Krasanakis
 * @see EmptyMatrix
 */
public class EmptyTensor extends Tensor {
	/**
	 * Initializes an {@link EmptyTensor} of zero size.
	 */
	public EmptyTensor() {
		super(0);
	}

	/**
	 * Initializes an {@link EmptyTensor} of the given size. It does not allocate
	 * memory for data.
	 * 
	 * @param size The tensor size.
	 */
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
