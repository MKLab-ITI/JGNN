package mklab.JGNN.models;

import java.util.List;

import mklab.JGNN.core.Tensor;

public interface TrainingData {
	public List<Tensor> getInputs();
	public List<Tensor> getOutputs();
}
