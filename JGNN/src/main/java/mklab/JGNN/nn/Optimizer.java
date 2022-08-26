package mklab.JGNN.nn;

import mklab.JGNN.core.Tensor;

/**
 * Provides an interface for training tensors. Has a {@link #reset()} method that starts potential training memory from scratch.
 * Has an {@link #update(Tensor, Tensor)} method that, given a current Tensor 
 * and a gradient operates on the former and adjusts its value.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract interface Optimizer {
	public void update(Tensor value, Tensor gradient);
	public default void reset() {};
}
