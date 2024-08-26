package mklab.JGNN.core.util;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Implements an iterator that traverses a range [min, max) where the right side
 * is non-inclusive. That is, this method behaves similarly to Python's
 * range(min, max). It is often used by {@link mklab.JGNN.core.Tensor} derived
 * classes to traverse through all element positions in sequential order.
 * 
 * @author Emmanouil Krasanakis
 */
public class Range implements Iterator<Long>, Iterable<Long> {
	private long nextValue;
	private final long max;

	/**
	 * Initializes a range [min, max) of subsequent integers where the right side is
	 * non-inclusive.
	 * 
	 * @param min The first value.
	 * @param max The value at which iteration stops (it is not reached).
	 */
	public Range(long min, long max) {
		this.nextValue = min;
		this.max = max;
	}

	@Override
	public boolean hasNext() {
		return nextValue < max;
	}

	@Override
	public Long next() {
		if (!hasNext())
			throw new NoSuchElementException();
		Long ret = Long.valueOf(nextValue);
		nextValue++;
		return ret;
	}

	@Override
	public Iterator<Long> iterator() {
		return this;
	}
}