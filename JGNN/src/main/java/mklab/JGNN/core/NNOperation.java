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
		ThreadData ret;
		synchronized(data) {
			ret = data.get(threadId);
		}
		if(ret==null) {
			synchronized(data) {
				data.put(threadId, ret = new ThreadData());
			}
		}return ret;
	}
	
	public void setDescription(String description) {
		this.description = description;
	}
	
	public String getDescription() {
		return description;
	}
	/**
	 * Retrieves an concise description of the operation that shows metadata
	 * and potential data descriptions processed by the current thread.
	 * @return A <code>String</code> description.
	 * @see #setDescription(String)
	 * @see #view()
	 */
	public String describe() {
		return this.getClass()+": "
					+(description!=null?description:("#"+this.hashCode()))+" = "
					+(data().lastOutput!=null?data().lastOutput.describe():"null");
	}
	
	/**
	 * Retrieves a string that views internal data being processed by the current thread,
	 * including gradients. This may
	 * @return A <code>String</code> view.
	 * @see #describe()
	 */
	public String view() {
		return this.getClass()+": "
					+describe()+" "
					+(data().lastOutput!=null?data().lastOutput.toString():"null")
					+" Delta"+(data().tapeError!=null?data().tapeError.toString():"null");
	}
	protected NNOperation() {}
	
	/**
	 * Retrieves a list of input operations within a model's execution graph.
	 * @return A list of {@link NNOperation}s.
	 */
	public ArrayList<NNOperation> getInputs() {
		return inputs;
	}
	
	/**
	 * Retrieves a list of output operations within a model's execution graph.
	 * @return A list of {@link NNOperation}s.
	 */
	public ArrayList<NNOperation> getOutputs() {
		return outputs;
	}
	
	/**
	 * Checks whether the operation yields a constant output.
	 * @return A <code>boolean</code> value.
	 */
	public boolean isConstant() {
		// TODO: automatically check inputs in the future
		return false;
	}
	
	/**
	 * Retrieves the degree of non-linearity of the operation
	 * to be used by {@link mklab.JGNN.initializers.VariancePreservingInitializer}.
	 * Default is one for operations like addition, multiplication, and matrix multiplication,
	 * and is different only for activation functions.
	 * @param inputId The input for which the non-linearity is calculated.
	 * @param inputMass The fraction of (matrix) parameters affecting the calculation coming from the respective input.
	 * @param outputNonLinearity The output's non-linearity gain.
	 * @return <code>double</code> describing the non-linearity.
	 */
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		return outputNonLinearity;
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
		try {
			ThreadData data = data();
			if(data.lastOutput!=null)
				return data.lastOutput;
			ArrayList<Tensor> lastInputs = new ArrayList<Tensor>();
			for(NNOperation input : inputs)
				lastInputs.add(input.runPrediction());
			/*System.out.println("Predicting... "+this.getClass());
			for(Tensor input : lastInputs)
				System.out.println("\t"+input.describe());
			if(data()!=data)
				System.out.println(data+" -> "+data());*/
			if(data()!=data)
				throw new RuntimeException("Thread data object should not change within the same thread");
			data.lastOutput = forward(lastInputs);
			data.tapeError = null;
			data.countTapeSources = 0;
			//System.out.println("Predicted "+describe());
			return data.lastOutput;
		}
		catch(Exception e) {
			System.err.println(e.toString());
			System.err.println("During the forward pass of "+describe()+" with the following inputs:");
			for(NNOperation input : inputs)
				System.err.println("\t"+input.describe());
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}
	
	final void backpropagate(Optimizer optimizer, Tensor error) {
		try {
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
		catch(Exception e) {
			System.err.println(e.toString());
			System.err.println("During the backward pass of "+describe()+" with derivative:");
			System.err.println("\t "+(error==null?"null":error.describe()));
			System.err.println("and the following inputs:");
			for(NNOperation input : inputs)
				System.err.println("\t"+input.describe());
			e.printStackTrace();
			System.exit(1);
		}
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
