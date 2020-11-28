package mklab.JGNN.core.util;

import java.util.AbstractMap;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Map.Entry;

public class Range2D implements Iterator<Entry<Long,Long>> {
	  private long nextValue;
	  private long nextValue2;
	  private final long min;
	  private final long max;
	  private final long max2;
	  public Range2D(long min, long max, long min2, long max2) {
	    this.nextValue = min;
	    this.nextValue2 = min2;
	    this.min = min;
	    this.max = max;
	    this.max2 = max2;
	  }
	  public boolean hasNext() {
	    return nextValue<max && nextValue2 < max2;
	  }
	  public Entry<Long,Long> next() {
	    if (!hasNext()) 
	      throw new NoSuchElementException();
	    Entry<Long,Long> ret = new AbstractMap.SimpleEntry<Long,Long>(Long.valueOf(nextValue), Long.valueOf(nextValue2));
	    nextValue++;
	    if(nextValue==max) {
	    	nextValue = min;
	    	nextValue2++;
	    }
	    return ret;
	  }
	  public void remove() {
	    throw new UnsupportedOperationException();
	  }
}