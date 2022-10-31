package mklab.JGNN.nn.inputs;

import mklab.JGNN.nn.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link mklab.JGNN.nn.NNOperation} that holds a constant tensor.
 * This value *is not affected by learning* but can be manually updated with the {@link #setTo(Tensor)} method.
 * 
 * @author Emmanouil Krasanakis
 */
public class Constant extends Parameter {
	/**
	 * Creates a constant holding a tensor.
	 * @param tensor The held tensor.
	 */
	public Constant(Tensor tensor) {
		super(tensor);
	}
	@Override
	public boolean isConstant() {
		return true;
	}
	@Override
	public boolean isCachable() {
		return tensor == data().lastOutput;
	}
	@Override
	protected void trainParameters(Optimizer optimizer, Tensor error) {
	}
}