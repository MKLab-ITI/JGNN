package mklab.JGNN.adhoc;

import java.util.List;

import mklab.JGNN.core.Tensor;

public class BatchData {
	private List<Tensor> inputs;
	private List<Tensor> outputs;
	public BatchData(List<Tensor> inputs, List<Tensor> outputs) {
		this.inputs = inputs;
		this.outputs = outputs;
	}
	public List<Tensor> getInputs() {
		return inputs;
	}
	public List<Tensor> getOutputs() {
		return outputs;
	}
}
