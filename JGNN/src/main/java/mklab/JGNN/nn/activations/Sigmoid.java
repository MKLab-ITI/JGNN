package mklab.JGNN.nn.activations;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Loss;
import mklab.JGNN.nn.inputs.Constant;

/**
 * Implements a {@link NNOperation} that performs a sigmoid transformation of its single input.
 * 
 * @author Emmanouil Krasanakis
 */
public class Sigmoid extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		return Loss.sigmoid(inputs.get(0));
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return Loss.sigmoidDerivative(inputs.get(0)).selfMultiply(error);
	}
	@Override
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		return outputNonLinearity;
	}
}