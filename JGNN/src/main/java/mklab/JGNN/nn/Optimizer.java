package mklab.JGNN.nn;

import mklab.JGNN.core.Tensor;

/**
 * Provides an interface for training tensors. Has a {@link #reset()} method
 * that starts potential training memory from scratch. Has an
 * {@link #update(Tensor, Tensor)} method that, given a current Tensor and a
 * gradient operates on the former and adjusts its value.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract interface Optimizer {
	/**
	 * In-place updates the value of a tensor given its gradient. Some optimizers
	 * (e.g. Adama) require the exact same tensor instance to be provided so as to
	 * keep track of its optimization progress. The library makes sure to keep this
	 * constraint.
	 * 
	 * @param value    The tensor to update.
	 * @param gradient The tensor's gradient.
	 */
	public void update(Tensor value, Tensor gradient);

	/**
	 * Resets (and lets the garbage collector free) optimizer memory. Should be
	 * called at the beginning of training (<b>not</b> after each epoch).
	 */
	public default void reset() {
	};
}
