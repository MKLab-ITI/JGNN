package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mklab.JGNN.core.inputs.Variable;
import mklab.JGNN.core.tensor.RepeatTensor;
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
	
	public ArrayList<Tensor> predict(Tensor... inputs) {
		return this.predict(Arrays.asList(inputs));
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
		ArrayList<Tensor> weights = new ArrayList<Tensor> ();
		for(Tensor desiredOutput : desiredOutputs)
			weights.add(new RepeatTensor(1, desiredOutput.size()));
		return trainSample(optimizer, inputs, desiredOutputs, weights);
	}
	
	public List<Tensor> trainSample(Optimizer optimizer, List<Tensor> inputs, List<Tensor> desiredOutputs, List<Tensor> weights) {
		if(inputs.size() != this.inputs.size())
			throw new RuntimeException("Incompatible number of inputs: "+inputs.size()+" given but "+this.inputs.size()+" expected");
		if(desiredOutputs.size() != this.outputs.size())
			throw new RuntimeException("Incompatible number of outputs: "+desiredOutputs.size()+" given but "+this.outputs.size()+" expected");
		ArrayList<Tensor> outputs = predict(inputs);
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i).zeroCopy();
			for(long pos : weights.get(i).getNonZeroElements()) 
				diff.put(pos, weights.get(i).get(pos)*Loss.crossEntropyDerivative(outputs.get(i).get(pos), desiredOutputs.get(i).get(pos)));
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		return outputs;
	}
	
	public double trainSample(Optimizer optimizer, List<Tensor> inputs) {
		if(inputs.size() != this.inputs.size())
			throw new RuntimeException("Incompatible number of inputs: "+inputs.size()+" but "+this.inputs.size()+" expected");
		ArrayList<Tensor> outputs = predict(inputs);
		double loss = 0;
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i);
			for(long pos : diff.getNonZeroElements())
				loss += Math.abs(diff.get(pos));
			for(long pos : diff.getNonZeroElements()) 
				diff.put(pos, outputs.get(i).get(pos));
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		return loss; 
	}
}
