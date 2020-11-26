package mklab.JGNN.core.operations.NN;

import java.util.List;

import mklab.JGNN.core.operations.NNOperation;
import mklab.JGNN.core.primitives.Tensor;

public class PRelu extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor x = inputs.get(0);
		Tensor param = inputs.get(1);
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements()) {
			double val = x.get(i);
			ret.put(i, val>0?val:(val*param.get(i)));
		}
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor x = inputs.get(0);
		Tensor param = inputs.get(1);
		Tensor ret = error.zeroCopy();
		if(inputId==0)
			for(long i : error.getNonZeroElements()) {
				double val = x.get(i);
				ret.put(i, val>=0?error.get(i):(error.get(i)*param.get(i)));
			}
		else
			for(long i : x.getNonZeroElements()) {
				double val = x.get(i);
				if(val<0)
					ret.put(i, error.get(i)*val);
			}
		return ret;
	}
}