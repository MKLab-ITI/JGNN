package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import mklab.JGNN.core.tensor.DenseTensor;

public class Slice implements Iterable<Long> {
	private List<Long> ids;
	
	public Slice(Iterable<Long> collection) {
		this.ids = new ArrayList<Long>();
		for(long id : collection)
			this.ids.add(id);
	}
	
	public Slice shuffle() {
		Collections.shuffle(ids);
		return this;
	}
	
	public Slice shuffle(int seed) {
		Collections.shuffle(ids, new Random(seed));
		return this;
	}
	
	public Slice range(int from, int end) {
		return new Slice(ids.subList(from, end));
	}
	
	public Slice range(double from, double end) {
		if(from<1)
			from = (int)(from*size());
		if(end<=1)
			end = (int)(end*size());
		return range((int)from, (int)end);
	}
	
	public int size() {
		return ids.size();
	}
	
	@Override
	public Iterator<Long> iterator() {
		return ids.iterator();
	}
	
	public Tensor asTensor() {
		return new DenseTensor(iterator());
	}

}
