package mklab.JGNN.nn.optimizers;

import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Wraps an {@link Optimizer} by applying the derivative of L2 loss
 * on every tensor during {@link Optimizer#update(Tensor, Tensor)}.
 * 
 * @author Emmanouil Krasanakis
 */
public class Regularization implements Optimizer {
	private Optimizer baseOptimizer;
	protected double regularization;
	/**
	 * Initializes a {@link Regularization}.
	 * @param baseOptimizer The base optimizer on which to apply regularization.
	 * @param regularization The weight of the regularization.
	 */
	public Regularization(Optimizer baseOptimizer, double regularization) {
		this.baseOptimizer = baseOptimizer;
		this.regularization = regularization;
	}
	protected Regularization() {}
	@Override
	public void update(Tensor value, Tensor gradient) {
		if(regularization==0)
			baseOptimizer.update(value, gradient);
		else
			baseOptimizer.update(value, gradient.add(value.multiply(regularization)));
	}
	@Override
	public void reset() {
		baseOptimizer.reset();
	}
}