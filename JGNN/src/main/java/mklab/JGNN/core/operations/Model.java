package mklab.JGNN.core.operations;

import java.util.ArrayList;
import java.util.List;

import mklab.JGNN.core.operations.NN.Variable;
import mklab.JGNN.core.primitives.Optimizer;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.util.Loss;

public class Model {
	private ArrayList<Variable> inputs = new ArrayList<Variable>();
	private ArrayList<NNOperation> outputs = new ArrayList<NNOperation>();
	
	public Model() {
	}
	public Model addInput(Variable input) {
		inputs.add(input);
		return this;
	}
	public Model addOutput(NNOperation output) {
		outputs.add(output);
		return this;
	}
	public ArrayList<Variable> getInputs() {
		return inputs;
	}
	public ArrayList<NNOperation> getOutputs() {
		return outputs;
	}
	
	public ArrayList<Tensor> predict(List<Tensor> inputs) {
		if(inputs.size() != this.inputs.size())
			throw new RuntimeException("Incompatible input size: expected"+this.inputs.size()+" inputs instead of "+inputs.size());
		for(NNOperation output : this.outputs)
			output.clearPrediction();
		for(int i=0;i<inputs.size();i++)
			this.inputs.get(i).setTo(inputs.get(i));
		ArrayList<Tensor> outputs = new ArrayList<Tensor> ();
		for(int i=0;i<this.outputs.size();i++)
			outputs.add(this.outputs.get(i).runPrediction());
		return outputs;
	}
	
	public List<Tensor> trainSampleDifference(Optimizer optimizer, List<Tensor> inputs, List<Tensor> desiredOutputs) {
		if(inputs.size() != this.inputs.size())
			throw new RuntimeException("Incompatible input size");
		if(desiredOutputs.size() != this.outputs.size())
			throw new RuntimeException("Incompatible output size");
		ArrayList<Tensor> outputs = predict(inputs);
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i).zeroCopy();
			for(long pos : outputs.get(i).getNonZeroElements()) 
				diff.put(pos, outputs.get(i).get(pos)-desiredOutputs.get(i).get(pos));
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		return outputs;
	}
	
	public List<Tensor> trainSample(Optimizer optimizer, List<Tensor> inputs, List<Tensor> desiredOutputs) {
		if(inputs.size() != this.inputs.size())
			throw new RuntimeException("Incompatible input size");
		if(desiredOutputs.size() != this.outputs.size())
			throw new RuntimeException("Incompatible output size");
		ArrayList<Tensor> outputs = predict(inputs);
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i).zeroCopy();
			for(long pos : diff.getNonZeroElements()) 
				diff.put(pos, Loss.crossEntropyDerivative(outputs.get(i).get(pos), desiredOutputs.get(i).get(pos)));
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		return outputs;
	}
	
	public double trainSample(Optimizer optimizer, List<Tensor> inputs) {
		if(inputs.size() != this.inputs.size())
			throw new RuntimeException("Incompatible input size");
		ArrayList<Tensor> outputs = predict(inputs);
		double loss = 0;
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i);
			for(long pos : diff.getNonZeroElements())
				loss += Math.abs(diff.get(pos));
			for(long pos : diff.getNonZeroElements()) 
				diff.put(pos, Loss.crossEntropyDerivative(outputs.get(i).get(pos), 0));
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		return loss; 
	}
}
