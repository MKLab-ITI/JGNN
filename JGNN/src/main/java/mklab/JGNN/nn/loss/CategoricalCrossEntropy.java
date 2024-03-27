package mklab.JGNN.nn.loss;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.Loss;

/**
 * Implements a categorical cross-entropy {@link Loss}.<br>
 * For binary classification of one output use {@link BinaryCrossEntropy}
 * @author Emmanouil Krasanakis
 */
public class CategoricalCrossEntropy extends Loss {
	private double epsilon;
	private boolean meanReduction;

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
	
	/**
	 * Sets the reduction mechanism of categorical cross entropy.
	 * This can be either a sum or a mean across the categorical cross entropy of all data samples.
	 * @param meanReduction true to perform mean reduction, false (default) for sum reduction.
	 * @return <code>this</code> CategoricalCrossEntropy object.
	 */
	public CategoricalCrossEntropy setMeanReduction(boolean meanReduction) {
		this.meanReduction = meanReduction;
		return this;
	}
	
	@Override
	public double evaluate(Tensor output, Tensor desired) {
		double ret = -output.add(epsilon).selfLog().selfMultiply(desired).sum();
		if(meanReduction)
			ret /= output.cast(Matrix.class).getRows();
		return ret;
	}
	
	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		Tensor ret = desired.multiply(output.add(epsilon).selfInverse()).negative();
		if(meanReduction)
			ret.selfMultiply(1. / output.cast(Matrix.class).getRows());
		return ret;
	}
}
