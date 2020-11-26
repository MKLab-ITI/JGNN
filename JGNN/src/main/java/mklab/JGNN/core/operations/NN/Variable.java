package mklab.JGNN.core.operations.NN;

import mklab.JGNN.core.primitives.Optimizer;
import mklab.JGNN.core.primitives.Tensor;

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