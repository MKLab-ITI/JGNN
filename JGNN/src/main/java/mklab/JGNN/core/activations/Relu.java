package mklab.JGNN.core.activations;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that performs a relu transformation of its single input first introduced by
 * <i>Hahnloser, Richard HR, Rahul Sarpeshkar, Misha A. Mahowald, Rodney J. Douglas, and H. Sebastian Seung.
 * "Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit."
 * Nature 405, no. 6789 (2000): 947-951.
 * </i>
 * @author Emmanouil Krasanakis
 */
public class Relu extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor x = inputs.get(0);
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements()) {
			double val = x.get(i);
			ret.put(i, val>0?val:0);
		}
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor x = inputs.get(0);
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements()) {
			double val = x.get(i);
			if(val>=0)
				ret.put(i, error.get(i));
		}
		return ret;
	}
}