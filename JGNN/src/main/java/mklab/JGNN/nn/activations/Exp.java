package mklab.JGNN.nn.activations;

import java.util.List;

import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that performs an exponential transformation of its single input.
 * 
 * @author Emmanouil Krasanakis
 */
public class Exp extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		return inputs.get(0).expMinusOne().selfAdd(1.);
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return output.multiply(error);
	}
	@Override
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		return outputNonLinearity;
	}
}