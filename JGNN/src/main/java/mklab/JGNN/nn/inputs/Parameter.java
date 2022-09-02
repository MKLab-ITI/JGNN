package mklab.JGNN.nn.inputs;

import java.util.List;

import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.nn.Optimizer;
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
	/**
	 * Gets sets the parameter's value tensor 
	 * @return The current value {@link Tensor}.
	 */
	public Tensor get() {
		return this.tensor;
	}
	@Override
	public NNOperation addInput(NNOperation inputComponent) {
		throw new RuntimeException("Parameter can not have inputs");
	}
	@Override
	protected void trainParameters(Optimizer optimizer, Tensor error) {
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
	@Override
	public boolean isConstant() {
		return false;
	}
	@Override
	public boolean isCachable() {
		return false;
	}
}