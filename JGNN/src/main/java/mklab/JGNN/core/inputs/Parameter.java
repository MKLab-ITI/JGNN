package mklab.JGNN.core.inputs;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that holds and returns a parameter tensor.
 * This value *is tuned by learning*.
 * 
 * @author Emmanouil Krasanakis
 */
public class Parameter extends NNOperation {
	protected Tensor tensor;
	public Parameter(Tensor tensor) {
		this.tensor = tensor;
		if(tensor!=null)
			runPrediction();
	}
	@Override
	public NNOperation addInput(NNOperation inputComponent) {
		throw new RuntimeException("Parameter can not have inputs");
	}
	@Override
	protected void trainParameters(Optimizer optimizer, Tensor error) {
		optimizer.update(tensor, error);
	}
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		return tensor;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return null;
	}
}