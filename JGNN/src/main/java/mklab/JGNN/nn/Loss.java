package mklab.JGNN.nn;

import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.core.Tensor;

/**
 * This class provides an abstract implementation of loss functions to be used
 * during {@link Model} training. Preferred use is by passing loss instances to
 * {@link ModelTraining}s.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class Loss {
	/**
	 * Provides a numerical evaluation of a loss function, so that lower values
	 * correspond to better predictions.
	 * 
	 * @param output  A model's estimation of true outputs.
	 * @param desired The expected outputs.
	 * @return A <code>double</code> value (is negative if smaller values are
	 *         better).
	 * @see #derivative(Tensor, Tensor)
	 */
	public abstract double evaluate(Tensor output, Tensor desired);

	/**
	 * Provides the derivative of a loss function at its evaluation point.
	 * 
	 * @param output  A model's estimation of true outputs.
	 * @param desired The expected outputs.
	 * @return A <code>Tensor</code> compliant to the model's estimation.
	 * @see #evaluate(Tensor, Tensor)
	 */
	public abstract Tensor derivative(Tensor output, Tensor desired);

	/**
	 * Called by {@link ModelTraining} instances to notify the loss that an epoch
	 * just ended. This is called only once per epoch and is where training progress
	 * should be reported to the user, if at all.
	 */
	public void onEndEpoch() {
	}

	/**
	 * Called by {@link ModelTraining} instance to notify the loss that a
	 * training loop just ended.
	 */
	public void onEndTraining() {
	}
}
