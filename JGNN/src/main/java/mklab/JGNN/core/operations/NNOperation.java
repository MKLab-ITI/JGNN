package mklab.JGNN.core.operations;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import mklab.JGNN.core.primitives.Optimizer;
import mklab.JGNN.core.primitives.Tensor;

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
		ThreadData ret = data.get(ThreadPool.getCurrentThreadId());
		if(ret==null) 
			data.put(ThreadPool.getCurrentThreadId(), ret = new ThreadData());
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

	synchronized public ArrayList<NNOperation> getInputs() {
		return inputs;
	}
	
	synchronized public ArrayList<NNOperation> getOutputs() {
		return outputs;
	}
	
	public boolean isConstant() {
		return false;
	}
	
	final void clearPrediction() {
		ThreadData data = data();
		if(data.lastOutput==null)
			return;
		data.lastOutput = null;
		for(NNOperation input : inputs)
			input.clearPrediction();
	}

	synchronized public NNOperation addInput(NNOperation inputComponent) {
		inputs.add(inputComponent);
		inputComponent.outputs.add(this);
		return this;
	}
	
	public final Tensor getLastTapeError() {
		return data().tapeError;
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
		data.tapeError = data.lastOutput.zeroCopy();
		data.countTapeSources = 0;
		//System.out.println("Predicted "+describe());
		return data.lastOutput;
	}
	
	final void backpropagate(Optimizer optimizer, Tensor error) {
		ThreadData data = data();
		//if(error!=null)
		//	System.out.println("Packpropagating... "+describe()+" Derivative "+error.describe());
		if(error!=null) 
			data.tapeError.selfAdd(error);
		data.countTapeSources++;
		if(data.countTapeSources>outputs.size())
			throw new RuntimeException("Redundant backpropagations were erroneously called");
		if(data.countTapeSources<outputs.size())
			return;
		if(error==null)
			return;
		//System.out.println("Packpropagating... "+describe()+" Derivative "+data.tapeError.describe());
		ArrayList<Tensor> lastInputs = new ArrayList<Tensor>();
		for(NNOperation input : inputs)
			lastInputs.add(input.data().lastOutput);
		for(int i=0;i<inputs.size();i++)
			if(!inputs.get(i).isConstant())
				inputs.get(i).backpropagate(optimizer, partial(i, lastInputs, data.lastOutput, data.tapeError));
		trainParameters(optimizer, data.tapeError);
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
	
	protected void trainParameters(Optimizer optimizer, Tensor error) {
	}
	protected abstract Tensor forward(List<Tensor> inputs);
	protected abstract Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error);
}
