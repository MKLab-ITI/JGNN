package mklab.JGNN.core.util;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Implements an iterator that traverses a range (similar to Python's range(min, max) method).
 * It is often used by {@link mklab.JGNN.core.Tensor} derived classes to traverse through all 
 * element positions in sequential order.
 * 
 * @author Emmanouil Krasanakis
 */
public class Range implements Iterator<Long>, Iterable<Long> {
	  private long nextValue;
	  private final long max;
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