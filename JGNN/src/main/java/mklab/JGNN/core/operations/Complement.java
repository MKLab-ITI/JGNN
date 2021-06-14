package mklab.JGNN.core.operations;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that performs the operation 1-x for its simple input x.
 * 
 * @author Emmanouil Krasanakis
 */
public class Complement extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		return inputs.get(0).multiply(-1).selfAdd(1);
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return error.multiply(-1);
	}
}
