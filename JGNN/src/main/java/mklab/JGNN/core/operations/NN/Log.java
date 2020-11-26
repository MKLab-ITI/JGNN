package mklab.JGNN.core.operations.NN;

import java.util.List;

import mklab.JGNN.core.operations.NNOperation;
import mklab.JGNN.core.primitives.Tensor;

public class Log extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor ret = inputs.get(0).zeroCopy();
		for(long i : inputs.get(0).getNonZeroElements())
			if(inputs.get(0).get(i)!=0)
				ret.put(i, Math.log(inputs.get(0).get(i)));
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return inputs.get(0).inverse().selfMultiply(error);
	}
}