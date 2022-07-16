package mklab.JGNN.nn.inputs;

import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link mklab.JGNN.core.NNOperation} that holds a constant tensor.
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
	protected void trainParameters(Optimizer optimizer, Tensor error) {
	}
}