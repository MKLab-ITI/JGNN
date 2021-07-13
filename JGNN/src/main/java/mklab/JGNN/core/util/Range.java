package mklab.JGNN.core.util;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * This class implements an iterator that traverses a range (similar to Python's range(min, max) method).
 * It is often used by {@link mklab.JGNN.core.Tensor} to traverse through all element positions.
 * 
 * @author Emmanouil Krasanakis
 */
public class Range implements Iterator<Long> {
	  private long nextValue;
	  private final long max;
	  public Range(long min, long max) {
	    this.nextValue = min;
	    this.max = max;
	  }
	  public boolean hasNext() {
	    return nextValue < max;
	  }
	  public Long next() {
	    if (!hasNext()) 
	      throw new NoSuchElementException();
	    Long ret = Long.valueOf(nextValue);
	    nextValue++;
	    return ret;
	  }
	  public void remove() {
	    throw new UnsupportedOperationException();
	  }
}