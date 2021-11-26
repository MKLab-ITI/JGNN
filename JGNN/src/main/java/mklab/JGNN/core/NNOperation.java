package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * This class defines an abstract neural network operation with forward and backpropagation capabilities.
 * Defined operations create execution trees based on input dependencies, which can then be run by
 * {@link Model} instances to make predictions. Creating the execution tree can be done by using
 * the {@link #addInput(NNOperation)} method. The correct number of inputs should be added to each operation.
 * Compliance to this rule needs to be checked by individual operations during forward passes.
 * <br>
 * Operations are thread-safe in the sense that they store gradients for backward passes on different
 * objects across different threads. This, way models can perform learning passes which are all synchronized
 * when eventually backpropagation feeds {@link mklab.JGNN.nn.inputs.Parameter} updates to an {@link Optimizer}.
 * <br>
 * The internal state of operations can be obtained with {@link #getPrediction()} to obtain their last
 * {@link Tensor} output (this output is depends on the thread calling the operation) and {@link #getLastTapeError()}
 * to obtain the last gradient obtained through backpropagation.
 *  
 * @author Emmanouil Krasanakis
 */
public abstract class NNOperation {
	private ArrayList<NNOperation> inputs = new ArrayList<NNOperation>();
	private ArrayList<NNOperation> outputs = new ArrayList<NNOperation>();
	private String description = null;
	
	protected static class ThreadData {
		public Tensor lastOutput;
		public Tensor tapeError;
		public int countTapeSources;
	}
	
	private HashMap<Integer, ThreadData> data = new HashMap<Integer, ThreadData>();
	
	protected ThreadData data() {
		int threadId = ThreadPool.getCurrentThreadId();
		ThreadData ret = data.get(threadId);
		if(ret==null) 
			data.put(threadId, ret = new ThreadData());
		return ret;
	}
	
	public void setDescription(String description) {
		this.description = description;
	}
	
	public String describe() {
		return this.getClass()+": "
					+(description!=null?description:this.hashCode())+" "
					+(data().lastOutput!=null?data().lastOutput.describe():"null");
	}
	public String view() {
		return this.getClass()+": "
					+describe()+" "
					+(data().lastOutput!=null?data().lastOutput.toString():"null")
					+" Delta"+(data().tapeError!=null?data().tapeError.toString():"null");
	}
	protected NNOperation() {}

	public ArrayList<NNOperation> getInputs() {
		return inputs;
	}
	
	public ArrayList<NNOperation> getOutputs() {
		return outputs;
	}
	
	public boolean isConstant() {
		return false;
	}
	
	public final void clearPrediction() {
		ThreadData data = data();
		if(data.lastOutput==null)
			return;
		data.lastOutput = null;
		for(NNOperation input : inputs)
			input.clearPrediction();
	}

	public NNOperation addInput(NNOperation inputComponent) {
		inputs.add(inputComponent);
		inputComponent.outputs.add(this);
		return this;
	}
	
	public final Tensor getLastTapeError() {
		return data().tapeError;
	}
	
	public final Tensor getPrediction() {
		return data().lastOutput;
	}
	
	public final Tensor runPrediction() {
		ThreadData data = data();
		if(data.lastOutput!=null)
			return data.lastOutput;
		ArrayList<Tensor> lastInputs = new ArrayList<Tensor>();
		for(NNOperation input : inputs)
			lastInputs.add(input.runPrediction());
		//System.out.println("Predicting... "+this.getClass());
		data.lastOutput = forward(lastInputs);
		data.tapeError = null;
		data.countTapeSources = 0;
		//System.out.println("Predicted "+describe());
		return data.lastOutput;
	}
	
	final void backpropagate(Optimizer optimizer, Tensor error) {
		ThreadData data = data();
		//if(error!=null)
		//	System.out.println("Packpropagating... "+describe()+" Derivative "+error.describe()+" on thread "+ThreadPool.getCurrentThreadId());
		if(error!=null) {
			if(data.tapeError==null)
				data.tapeError = data.lastOutput.zeroCopy();
			data.tapeError.selfAdd(error);
		}
		data.countTapeSources++;
		if(data.countTapeSources>outputs.size())
			throw new RuntimeException("Redundant backpropagations were erroneously called");
		if(data.countTapeSources<outputs.size())
			return;
		if(error==null)
			return;
		//System.out.println("Packpropagating... "+describe()+" Derivative "+data.tapeError+" prev out "+data.lastOutput);
		ArrayList<Tensor> lastInputs = new ArrayList<Tensor>();
		for(NNOperation input : inputs)
			lastInputs.add(input.data().lastOutput);
		for(int i=0;i<inputs.size();i++)
			if(!inputs.get(i).isConstant())
				inputs.get(i).backpropagate(optimizer, partial(i, lastInputs, data.lastOutput, data.tapeError));
		trainParameters(optimizer, data.tapeError);
		//System.out.println("Finished backpropagation on "+describe()+" on thread "+ThreadPool.getCurrentThreadId());
	}

	final void forceBackpropagate(Optimizer optimizer, Tensor error) {
		ThreadData data = data();
		data.tapeError = error;
		ArrayList<Tensor> lastInputs = new ArrayList<Tensor>();
		for(NNOperation input : inputs)
			lastInputs.add(input.data().lastOutput);
		for(int i=0;i<inputs.size();i++)
			inputs.get(i).backpropagate(optimizer, partial(i, lastInputs, data.lastOutput, error));
		trainParameters(optimizer, error);
	}

	/** 
	 * Performs a forward pass in the operation <b>without inducing any kind of learning or storing the outcome</b>.
	 * This is just a way to replicate the operation at the tensor level and does not affect or is affected by
	 * any dependent inputs {@link #addInput}.
	 * @param inputs A list of input tensors needed by the operation.
	 * @return A Tensor with the operation's outcome.
	 */
	public final Tensor run(List<Tensor> inputs) {
		return forward(inputs);
	}

	/** 
	 * Performs a forward pass in the operation <b>without inducing any kind of learning or storing the outcome</b>.
	 * This is just a way to replicate the operation at the tensor level and does not affect or is affected by
	 * any dependent inputs {@link #addInput}.
	 * @param inputs A list of input tensors needed by the operation.
	 * @return A Tensor with the operation's outcome.
	 */
	public final Tensor run(Tensor... inputs) {
		return forward(Arrays.asList(inputs));
	}
	
	protected void trainParameters(Optimizer optimizer, Tensor error) {
	}
	protected abstract Tensor forward(List<Tensor> inputs);
	protected abstract Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error);
}
