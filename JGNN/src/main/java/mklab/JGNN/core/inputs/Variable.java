package mklab.JGNN.core.inputs;

import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Optimizer;
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
	public boolean isConstant() {
		return true;
	}
	@Override
	protected void trainParameters(Optimizer optimizer, Tensor error) {
	}
	public void setTo(Tensor value) {
		this.tensor = value;
	}
}