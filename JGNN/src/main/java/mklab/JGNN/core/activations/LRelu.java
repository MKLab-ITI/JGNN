package mklab.JGNN.core.activations;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that performs a leaky relu operation, where the first argument is a tensor on which
 * it is applied and the second one should be a tensor wrapping a double value (consider initializing this with as a 
 * {@link mklab.JGNN.core.operations.Constant} holding a tensor generated with {@link Tensor#fromDouble(double)}) where
 * the wrapped value indicates the negative region's slope. If the negative slope is zero, leaky relu is reduced to {@link Relu}.
 * 
 * @author Emmanouil Krasanakis
 */
public class LRelu extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		Tensor x = inputs.get(0);
		Tensor ret = x.zeroCopy();
		double mult = inputs.get(1).toDouble();
		for(long i : x.getNonZeroElements()) {
			double val = x.get(i);
			ret.put(i, val>0?val:(val*mult));
		}
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor x = inputs.get(0);
		Tensor ret = x.zeroCopy();
		double mult = inputs.get(1).toDouble();
		for(long i : x.getNonZeroElements()) {
			double val = x.get(i);
			if(val>=0)
				ret.put(i, error.get(i));
			else
				ret.put(i, mult*error.get(i));
		}
		return ret;
	}
}