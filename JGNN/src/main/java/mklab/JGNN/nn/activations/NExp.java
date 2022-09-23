package mklab.JGNN.nn.activations;

import java.util.List;

import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that performs an exponential transformation of
 * its single input, but only on the non-zero elements.
 * 
 * @author Emmanouil Krasanakis
 */
public class NExp extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor input = inputs.get(0);
		Tensor ret = input.zeroCopy();
		for(long pos : input)
			if(input.get(pos)!=0)
				ret.put(pos, Math.exp(input.get(pos)));
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor ret = output.copy();
		Tensor input = inputs.get(0);
		for(long pos : input)
			if(input.get(pos)!=0)
				ret.put(pos, Math.exp(input.get(pos)*error.get(pos)));
		return output.multiply(error);
	}
	@Override
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		return outputNonLinearity;
	}
}