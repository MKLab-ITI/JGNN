package mklab.JGNN.core.operations.NN;

import java.util.List;

import mklab.JGNN.core.operations.NNOperation;
import mklab.JGNN.core.primitives.Tensor;

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
