package mklab.JGNN.core.inputs;

import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;

public class Constant extends Parameter {
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
	public void setTo(Tensor value) {
		this.tensor = value;
	}
}