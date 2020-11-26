package mklab.JGNN.core.primitives;

/**
 * Provides an interface for training tensors. Has a {@link #reset()} method that starts potential training memory from scratch.
 * Has an {@link #update(Tensor, Tensor)} method that, given a current value and a gradient Tensor adjusts the tensor's value.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract interface Optimizer {
	public void update(Tensor value, Tensor gradient);
	public default void reset() {};
}
