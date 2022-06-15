package mklab.JGNN.nn.inputs;

import java.util.List;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that holds and returns a parameter tensor.
 * <b>The held value is tuned by learning.</b>
 * 
 * @author Emmanouil Krasanakis
 */
public class Parameter extends NNOperation {
	protected Tensor tensor;
	protected double regularization;
	public Parameter(Tensor tensor) {
		this(tensor, 0);
	}
	public Parameter(Tensor tensor, double regularization) {
		this.tensor = tensor;
		this.regularization = regularization;
		if(tensor!=null)
			runPrediction();
	}
	/**
	 * Forcefully sets the parameter's value tensor to the desired value.
	 * @param tensor The new parameter value.
	 * @return <code>this</code> parameter.
	 */
	public Parameter set(Tensor tensor) {
		this.tensor = tensor;
		return this;
	}
	@Override
	public NNOperation addInput(NNOperation inputComponent) {
		throw new RuntimeException("Parameter can not have inputs");
	}
	@Override
	protected void trainParameters(Optimizer optimizer, Tensor error) {
		//System.out.println(error.norm());
		if(regularization!=0)
			error = error.add(tensor.multiply(regularization));
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