package mklab.JGNN.nn.operations;

import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.NNOperation;

public class Reduce extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		Matrix edgeFeats = inputs.get(0).cast(Matrix.class);
		Matrix adj = inputs.get(1).cast(Matrix.class);
		Matrix ret = edgeFeats.zeroCopy(adj.getRows(), edgeFeats.getCols());
		long id = 0;
		for(Entry<Long, Long> entry : adj.getNonZeroEntries()) {
			ret.accessRow(entry.getKey()).selfAdd(edgeFeats.accessRow(id), adj.get(entry.getKey(), entry.getValue()));
			id += 1;
		}
		return ret;
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		if(inputId==1)
			throw new RuntimeException("Cannot backpropagate over adjacency matrices in reduce");
		Matrix edgeFeats = inputs.get(0).cast(Matrix.class);
		Matrix adj = inputs.get(1).cast(Matrix.class);
		Matrix err = error.cast(Matrix.class);
		Matrix ret = edgeFeats.zeroCopy();
		long id = 0;
		for(Entry<Long, Long> entry : adj.getNonZeroEntries()) {
			ret.accessRow(id).selfAdd(err.accessRow(entry.getKey()), adj.get(entry.getKey(), entry.getValue()));
			id += 1;
		}
		return ret;
	}

}
