package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import mklab.JGNN.core.tensor.RepeatTensor;
import mklab.JGNN.core.util.Loss;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.inputs.Variable;
import mklab.JGNN.nn.operations.Dropout;

/**
 * This class is a way to organize {@link NNOperation} trees into trainable machine
 * learning models. Critically, only model inputs and outputs need to be defined. It also
 * provides methods that perform training by calling forward and backward passes.
 * Models can have multiple inputs and outputs.
 * 
 * @author Emmanouil Krasanakis
 */
public class Model {
	private ArrayList<Variable> inputs = new ArrayList<Variable>();
	private ArrayList<NNOperation> outputs = new ArrayList<NNOperation>();
	
	public Model() {
	}
	
	public ArrayList<Parameter> getParameters(){
		ArrayList<Parameter> parameters = new ArrayList<Parameter>();
		ArrayList<NNOperation> pending = new ArrayList<NNOperation>();
		HashSet<NNOperation> visited = new HashSet<NNOperation>();
		for(NNOperation output : outputs) {
			visited.add(output);
			pending.add(output);
		}
		while(!pending.isEmpty()) {
			NNOperation operation = pending.remove(pending.size()-1);
			if(operation instanceof Parameter)
				parameters.add((Parameter) operation);
			for(NNOperation input : operation.getInputs()) 
				if(!visited.contains(input)){
					visited.add(input);
					pending.add(input);
				}
		}
		return parameters;
	}
	

	public Model setTraining(boolean training){
		ArrayList<NNOperation> pending = new ArrayList<NNOperation>();
		HashSet<NNOperation> visited = new HashSet<NNOperation>();
		for(NNOperation output : outputs) {
			visited.add(output);
			pending.add(output);
		}
		while(!pending.isEmpty()) {
			NNOperation operation = pending.remove(pending.size()-1);
			if(operation instanceof Dropout)
				((Dropout) operation).setEnabled(training);
			for(NNOperation input : operation.getInputs()) 
				if(!visited.contains(input)){
					visited.add(input);
					pending.add(input);
				}
		}
		return this;
	}
	
	/**
	 * Adds to the model's inputs the provided {@link Variable}.
	 * @param input A variable to set as an input.
	 * @return <code>this</code> Model instance.
	 * @see #addOutput(NNOperation)
	 * @see #getInputs()
	 * @see #predict(List)
	 * @see #predict(Tensor...)
	 */
	public Model addInput(Variable input) {
		inputs.add(input);
		return this;
	}
	
	/**
	 * Adds to the model's output the output of the provided operation.
	 * @param output An operation to set as an output.
	 * @return <code>this</code> Model instance.
	 * @see #addInput(Variable)
	 * @see #getOutputs()
	 * @see #predict(List)
	 * @see #predict(Tensor...)
	 */
	public Model addOutput(NNOperation output) {
		outputs.add(output);
		return this;
	}
	
	/**
	 * Retrieves a list of model inputs. Editing this list affects
	 * the model and is not recommended. Input order is based on
	 * the chronological addition of inputs through {@link #addInput(Variable)}.
	 * @return A list of {@link Variable} instances.
	 * @see #getOutputs()
	 */
	public ArrayList<Variable> getInputs() {
		return inputs;
	}
	
	/**
	 * Retrieves a list of model outputs. Editing this list affects
	 * the model and is not recommended. Output order is based on
	 * the chronological addition of outputs through {@link #addOutput(NNOperation)}.
	 * @return A list of {@link Variable} instances.
	 * @see #getInputs()
	 * @see #addOutput(NNOperation)
	 */
	public ArrayList<NNOperation> getOutputs() {
		return outputs;
	}
	
	/**
	 * Forward run of the model given an array of input tensors. 
	 * Wraps {@link #predict(List)}.
	 * @param inputs Input tensors to be assigned to input variables.
	 * @return A list of tensors output by the model after a forward pass.
	 * @see #predict(List)
	 */
	public ArrayList<Tensor> predict(Tensor... inputs) {
		return this.predict(Arrays.asList(inputs));
	}
	
	/**
	 * Forward run of the model given a list of input tensors. Their order should match the order
	 * of variables in {@link #getInputs()}.
	 * @param inputs A list of tensors to be assigned to input variables. These should have
	 * @return A list of tensors output by the model after a forward pass.
	 * @see #predict(Tensor...)
	 * @throws IllegalArgumentException if the number of input tensors does not match the number of input variables.
	 */
	public ArrayList<Tensor> predict(List<Tensor> inputs) {
		if(inputs.size() != this.inputs.size())
			throw new IllegalArgumentException("Incompatible input size: expected"+this.inputs.size()+" inputs instead of "+inputs.size());
		for(NNOperation output : this.outputs)
			output.clearPrediction();
		for(int i=0;i<inputs.size();i++)
			this.inputs.get(i).setTo(inputs.get(i));
		ArrayList<Tensor> outputs = new ArrayList<Tensor> ();
		for(int i=0;i<this.outputs.size();i++)
			outputs.add(this.outputs.get(i).runPrediction());
		return outputs;
	}
	
	/**
	 * Performs one parameter adjustment step (e.g. epoch) using {@link Optimizer} for a square loss function
	 * that compares desired outputs and the ones {@link #predict(List)} yields for the given inputs.
	 * @param optimizer The provided optimizer with which to adjust values.
	 * @param inputs A list of input tensors that would be passed to a corresponding {@link #predict(List)} call.
	 * @param desiredOutputs A list of output tensors that would be ideally returned by the prediction.
	 * @return A list of prediction outputs (the ones computed before parameter adjustment)
	 * @throws IllegalArgumentException If the number of inputs and outputs do not match the sizes of {@link #getInputs()}
	 * and {@link #getOutputs()} respectively.
	 * @see #trainCrossEntropy(Optimizer, List, List, List)
	 * @see #trainTowardsZero(Optimizer, List)
	 */
	public List<Tensor> trainL2(Optimizer optimizer, List<Tensor> inputs, List<Tensor> desiredOutputs) {
		if(inputs.size() != this.inputs.size())
			throw new IllegalArgumentException("Incompatible input size");
		if(desiredOutputs.size() != this.outputs.size())
			throw new IllegalArgumentException("Incompatible output size");
		setTraining(true);
		ArrayList<Tensor> outputs = predict(inputs);
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i).multiply(-1).selfAdd(desiredOutputs.get(i)).selfMultiply(-1);
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		setTraining(false);
		return outputs;
	}
	
	/**
	 * Performs the cross entropy training of {@link #trainCrossEntropy(Optimizer, List, List, List)} for unit weights.
	 * @param optimizer The provided optimizer with which to adjust values.
	 * @param inputs A list of input tensors that would be passed to a corresponding {@link #predict(List)} call.
	 * Element values should be either 1 or 0.
	 * @param desiredOutputs A list of output tensors that would be ideally returned by the prediction.
	 * Element values should lie in the rage [0,1].
	 * @return A list of prediction outputs (the ones computed before parameter adjustment)
	 * @throws IllegalArgumentException If the number of inputs and desired outputs do not match the sizes of {@link #getInputs()}
	 * and {@link #getOutputs()} respectively, or if the number of weight tensor do not match the number of desired outputs.
	 */
	public List<Tensor> trainCrossEntropy(Optimizer optimizer, List<Tensor> inputs, List<Tensor> desiredOutputs) {
		ArrayList<Tensor> weights = new ArrayList<Tensor> ();
		for(Tensor desiredOutput : desiredOutputs)
			weights.add(new RepeatTensor(1, desiredOutput.size()));
		return trainCrossEntropy(optimizer, inputs, desiredOutputs, weights);
	}

	/**
	 * Performs one parameter adjustment step (e.g. epoch) using {@link Optimizer} for a cross entropy loss function
	 * that compares desired outputs and the ones {@link #predict(List)} yields for the given inputs.
	 * @param optimizer The provided optimizer with which to adjust values.
	 * @param inputs A list of input tensors that would be passed to a corresponding {@link #predict(List)} call.
	 * Element values should be either 1 or 0.
	 * @param desiredOutputs A list of output tensors that would be ideally returned by the prediction.
	 * Element values should lie in the rage [0,1].
	 * @param weights A list of weight tensors to be applied element-by-element on the outcome of 
	 * {@link Loss#crossEntropyDerivative(double, double)}.
	 * @return A list of prediction outputs (the ones computed before parameter adjustment)
	 * @throws IllegalArgumentException If the number of inputs and desired outputs do not match the sizes of {@link #getInputs()}
	 * and {@link #getOutputs()} respectively, or if the number of weight tensor do not match the number of desired outputs.
	 * @see #trainCrossEntropy(Optimizer, List, List)
	 * @see #trainTowardsZero(Optimizer, List)
	 */
	public List<Tensor> trainCrossEntropy(Optimizer optimizer, List<Tensor> inputs, List<Tensor> desiredOutputs, List<Tensor> weights) {
		if(inputs.size() != this.inputs.size())
			throw new IllegalArgumentException("Incompatible number of inputs: "+inputs.size()+" given but "+this.inputs.size()+" expected");
		if(desiredOutputs.size() != this.outputs.size())
			throw new IllegalArgumentException("Incompatible number of outputs: "+desiredOutputs.size()+" given but "+this.outputs.size()+" expected");
		setTraining(true);
		ArrayList<Tensor> outputs = predict(inputs);
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i).zeroCopy();
			for(long pos : weights.get(i).getNonZeroElements()) 
				diff.put(pos, weights.get(i).get(pos)*Loss.crossEntropyDerivative(outputs.get(i).get(pos), desiredOutputs.get(i).get(pos)));
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		setTraining(false);
		return outputs;
	}
	
	/**
	 * Is equivalent to calling {@link #trainL2(Optimizer, List, List)} for all desired outputs set to zero. 
	 * Use this to train towards optimizing an explicitly defined loss function.
	 * @param optimizer The provided optimizer with which to adjust values.
	 * @param inputs A list of input tensors that would be passed to a corresponding {@link #predict(List)} call.
	 * @return The L2 loss (computed before parameter adjustment)
	 * @throws IllegalArgumentException If the number of inputs and outputs do not match the sizes of {@link #getInputs()}
	 * and {@link #getOutputs()} respectively.
	 * @see #trainCrossEntropy(Optimizer, List, List, List)
	 */
	public double trainTowardsZero(Optimizer optimizer, List<Tensor> inputs) {
		if(inputs.size() != this.inputs.size())
			throw new IllegalArgumentException("Incompatible number of inputs: "+inputs.size()+" but "+this.inputs.size()+" expected");
		setTraining(true);
		ArrayList<Tensor> outputs = predict(inputs);
		double loss = 0;
		for(int i=0;i<outputs.size();i++) {
			Tensor diff = outputs.get(i).abs();
			this.outputs.get(i).forceBackpropagate(optimizer, diff);
		}
		setTraining(false);
		return loss; 
	}
}
