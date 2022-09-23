package mklab.JGNN.nn.loss;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.Loss;

/**
 * Implements a categorical cross-entropy {@link Loss}.<br>
 * For binary classification of one output use {@link BinaryCrossEntropy}
 * @author Emmanouil Krasanakis
 */
public class CategoricalCrossEntropy extends Loss {
	private double epsilon;

	/**
	 * Initializes categorical cross entropy with 1.E-12 epsilon value.
	 * @see #BinaryCrossEntropy(double)
	 */
	public CategoricalCrossEntropy() {
		this(1.E-12);
	}

	/**
	 * Initializes categorical cross entropy with and epsilon value 
	 * to bound its outputs in the range [log(epsilon), -log(epsilon)] instead of (-inf, inf).
	 * @param epsilon A very small positive <code>double</code>.
	 */
	public CategoricalCrossEntropy(double epsilon) {
		this.epsilon = epsilon;
	}
	
	@Override
	public double evaluate(Tensor output, Tensor desired) {
		return -output.add(epsilon).selfLog().selfMultiply(desired).sum();// / output.cast(Matrix.class).getRows();
	}
	
	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		return desired.multiply(output.add(epsilon).selfInverse()).negative();//.selfMultiply(-1. / output.cast(Matrix.class).getRows());
	}
}
