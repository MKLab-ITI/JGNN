package mklab.JGNN.core.util;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class Range implements Iterator<Long> {
	  private long nextValue;
	  private final long max;
	  public Range(long min, long max) {
	    this.nextValue = min;
	    this.max = max;
	  }
	  public boolean hasNext() {
	    return nextValue < max-1;
	  }
	  public Long next() {
	    if (!hasNext()) 
	      throw new NoSuchElementException();
	    nextValue++;
	    return Long.valueOf(nextValue);
	  }
	  public void remove() {
	    throw new UnsupportedOperationException();
	  }
}