package mklab.JGNN.nn.pooling;

import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.NNOperation;

public class Sort extends NNOperation {
	private int k = 0;
	private String name = null;
	private long[] cache = null;
	
	public Sort(int k) {
		super();
		this.k = k;
		cache = new long[k];
	}
	
	public Sort setDimensionName(String name) {
		this.name = name;
		return this;
	}
	
	protected boolean compare(long pos1, long pos2, Tensor indexes, Matrix values) {
		pos1 = (long)indexes.get(pos1);
		pos2 = (long)indexes.get(pos2);
		long col = values.getCols()-1;
		while(col>0 && values.get(pos1, col)==values.get(pos2, col))
			col -= 1;
		return values.get(pos1, col) < values.get(pos2, col);
	}
	
	protected void merge(Tensor indexes, Matrix values, long from, long to, long middle) {
		long from2 = middle + 1;
		int pos = 0;
		long k = Math.min(this.k, to-from+1);
		long originalFrom = from;
		if(compare(middle, from2, indexes, values))
			return;
		//System.out.println("merging\n  "+indexes.accessSubtensor(0, middle+1).toString()+"\n  "+indexes.accessSubtensor(middle+1));
		while(pos<k) {
			if(from<=middle && (from2>to || compare(from, from2, indexes, values))) {
				cache[pos] = (long) indexes.get(from);
				from += 1;
			}
			else {
				cache[pos] = (long) indexes.get(from2);
				from2 += 1;
			}
			pos += 1;
		}
		for(int i=0;i<pos;i++)
			indexes.put(originalFrom+i, cache[i]);
	}
	
	protected void sort(Tensor indexes, Matrix values, long from, long to) {
		if(from>=to)
			return;
		long middle = (from+to)/2;
		sort(indexes, values, from, middle);
		sort(indexes, values, middle+1, to);
		merge(indexes, values, from, to, middle);
	}

	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Matrix input = inputs.get(0).cast(Matrix.class);
		Tensor order = Tensor.fromRange(input.getRows());
		//System.out.println(input.toString());
		sort(order, input, 0, input.getRows()-1);
		//System.out.println(order);
		return order.accessSubtensor(0, k).setDimensionName(name);
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		throw new RuntimeException("Sorting produces indices and should only be used in gather [...] statements");
	}

}
