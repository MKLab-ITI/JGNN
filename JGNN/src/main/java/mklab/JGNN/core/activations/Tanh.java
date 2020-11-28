package mklab.JGNN.core.activations;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Loss;

public class Tanh extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		return Loss.tanh(inputs.get(0));
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return Loss.tanhDerivative(inputs.get(0)).selfMultiply(error);
	}
}
