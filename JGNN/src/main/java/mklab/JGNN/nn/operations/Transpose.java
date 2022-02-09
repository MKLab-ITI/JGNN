package mklab.JGNN.nn.operations;

import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that performs matrix transposition.
 * 
 * @author Emmanouil Krasanakis
 */
public class Transpose extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		return ((Matrix)inputs.get(0)).asTransposed();
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return ((Matrix)error).asTransposed();
	}
}
