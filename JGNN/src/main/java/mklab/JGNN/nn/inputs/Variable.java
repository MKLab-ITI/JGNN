package mklab.JGNN.nn.inputs;

import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.nn.Optimizer;

import java.util.HashMap;
import java.util.List;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.ThreadPool;

/**
 * Implements a {@link NNOperation} that represents {@link mklab.JGNN.nn.Model} inputs.
 * Its values can be set using the {@link #setTo(Tensor)} method.
 * 
 * @author Emmanouil Krasanakis
 */
public class Variable extends NNOperation {
	private HashMap<Integer, Tensor> threadData = new HashMap<Integer, Tensor>();
	public Variable() {
	}
	
	@Override
	protected void trainParameters(Optimizer optimizer, Tensor error) {
	}
	
	public void setTo(Tensor value) {
		synchronized(threadData) {
			threadData.put(ThreadPool.getCurrentThreadId(), value);
		}
	}
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor ret;
		synchronized(threadData) {
			ret = threadData.get(ThreadPool.getCurrentThreadId());
		}
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		return null;
	}
	
	@Override
	public boolean isConstant() {
		return false;
	}
	@Override
	public boolean isCachable() {
		return false;
	}
}