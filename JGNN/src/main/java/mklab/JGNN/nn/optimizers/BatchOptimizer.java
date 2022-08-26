package mklab.JGNN.nn.optimizers;

import java.util.ArrayList;
import java.util.HashMap;

import mklab.JGNN.nn.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Wraps an {@link Optimizer} by accumulating derivatives and calling
 * {@link Optimizer#update(Tensor, Tensor)} with the average derivative
 * after a fixed number of accumulations. Accumulation restarts after
 * update. Provides a method {@link #updateAll()} to update all accumulated 
 * derivatives, for example in case the constructor {@link #BatchOptimizer(Optimizer)}
 * is used without inputting a fixed number of derivative updates.
 * 
 * @author Emmanouil Krasanakis
 */
public class BatchOptimizer implements Optimizer {
	private Optimizer baseOptimizer;
	private long batchSize;
	private HashMap<Tensor, Tensor> derivativeAccumulation = new HashMap<Tensor, Tensor>();
	private HashMap<Tensor, Integer> batchProgress = new HashMap<Tensor, Integer>();
	
	/**
	 * Initializes a {@link BatchOptimizer} that accumulates derivatives and updates them
	 * only when {@link #updateAll()} is called.
	 * @param baseOptimizer The base optimizer with which to perform the derivative updates.
	 */
	public BatchOptimizer(Optimizer baseOptimizer) {
		this.baseOptimizer = baseOptimizer;
		this.batchSize = Long.MAX_VALUE;
	}
	/**
	 * Initializes a {@link BatchOptimizer} that accumulates derivatives and updates them
	 * after a fixed number of updates.
	 * @param baseOptimizer The base optimizer with which to perform the derivative updates.
	 * @param batchSize The number of updates at which to pass the average accumulation to the base optimizer.
	 */
	public BatchOptimizer(Optimizer baseOptimizer, long batchSize) {
		this.baseOptimizer = baseOptimizer;
		this.batchSize = batchSize;
	}
	public void updateAll() {
		for(Tensor value : new ArrayList<Tensor>(derivativeAccumulation.keySet())) 
			synchronized(value) {
				if(batchProgress.get(value)!=0) {
					baseOptimizer.update(value, derivativeAccumulation.get(value).selfMultiply(1./batchProgress.get(value)));
					derivativeAccumulation.remove(value);
					batchProgress.remove(value);
				}
			}
	}
	@Override
	public void update(Tensor value, Tensor gradient) {
		synchronized(value) {
			if(!derivativeAccumulation.containsKey(value))
				derivativeAccumulation.put(value, value.zeroCopy());
			derivativeAccumulation.get(value).selfAdd(gradient);
			batchProgress.put(value, batchProgress.getOrDefault(value, 0)+1);
			if(batchProgress.get(value)>=batchSize) {
				baseOptimizer.update(value, derivativeAccumulation.get(value).selfMultiply(1./batchProgress.get(value)));
				derivativeAccumulation.remove(value);
				batchProgress.remove(value);
			}
		}
	}
	@Override
	public void reset() {
		derivativeAccumulation = new HashMap<Tensor, Tensor>();
		batchProgress = new HashMap<Tensor, Integer>();
		baseOptimizer.reset();
	}
}
