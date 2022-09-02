package mklab.JGNN.nn.inputs;

import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.nn.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that represents {@link mklab.JGNN.core.Model} inputs.
 * Its values can be set using the {@link #setTo(Tensor)} method.
 * 
 * @author Emmanouil Krasanakis
 */
public class Variable extends Parameter {
	public Variable() {
		super(null);
	}
	@Override
	protected void trainParameters(Optimizer optimizer, Tensor error) {
	}
	@Override
	public boolean isConstant() {
		return true;
	}
	@Override
	public boolean isCachable() {
		return false;
	}
	public void setTo(Tensor value) {
		this.tensor = value;
	}
}