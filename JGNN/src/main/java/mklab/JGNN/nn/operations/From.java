package mklab.JGNN.nn.operations;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;

/**
 * Implements a {@link NNOperation} that lists the first element of the 2D matrix element iterator.
 * 
 * @author Emmanouil Krasanakis
 */
public class From extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		ArrayList<Long> ret = new ArrayList<Long>((int) inputs.get(0).estimateNumNonZeroElements());
		for(Entry<Long, Long> entry : inputs.get(0).cast(Matrix.class).getNonZeroEntries())
			ret.add(entry.getKey());
		return new DenseTensor(ret.iterator());
	}
	
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		throw new UnsupportedOperationException("Cannot iterate over non-constant matrices");
	}
}