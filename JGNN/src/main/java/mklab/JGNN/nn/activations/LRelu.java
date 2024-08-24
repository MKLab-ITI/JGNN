package mklab.JGNN.nn.activations;

import java.util.List;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.nn.inputs.Constant;

/**
 * Implements a {@link NNOperation} that performs a leaky relu operation, where
 * the first argument is a tensor on which it is applied and the second one
 * should be a tensor wrapping a double value (consider initializing this with
 * as a {@link mklab.JGNN.nn.inputs.Constant} holding a tensor generated with
 * {@link Tensor#fromDouble(double)}) where the wrapped value indicates the
 * negative region's slope. If the negative slope is zero, leaky relu is reduced
 * to {@link Relu}.
 * 
 * @author Emmanouil Krasanakis
 */
public class LRelu extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if (inputs.size() != 2)
			throw new IllegalArgumentException();
		Tensor x = inputs.get(0);
		Tensor ret = x.zeroCopy();
		double mult = inputs.get(1).toDouble();
		for (long i : x.getNonZeroElements()) {
			double val = x.get(i);
			ret.put(i, val > 0 ? val : (val * mult));
		}
		return ret;
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor x = inputs.get(0);
		Tensor ret = x.zeroCopy();
		double mult = inputs.get(1).toDouble();
		for (long i : x.getNonZeroElements()) {
			double val = x.get(i);
			if (val >= 0)
				ret.put(i, error.get(i));
			else
				ret.put(i, mult * error.get(i));
		}
		return ret;
	}

	@Override
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		double slope = ((Constant) getInputs().get(1)).get().toDouble();
		return outputNonLinearity * Math.sqrt(2. / (1 + (slope * slope)));
	}

}