package mklab.JGNN.nn.operations;

import java.util.List;

import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that just transfers its single input.
 * 
 * @author Emmanouil Krasanakis
 */
public class Identity extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		return inputs.get(0);
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return error;
	}
}