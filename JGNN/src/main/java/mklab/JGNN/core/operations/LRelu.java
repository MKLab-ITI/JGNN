package mklab.JGNN.core.operations;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;

public class LRelu extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
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