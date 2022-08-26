package mklab.JGNN.core.loss;

import mklab.JGNN.core.Loss;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Implements a categorical cross-entropy {@link Loss}.<br>
 * For binary classification of one output use {@link BinaryCrossEntropy}
 * @author Emmanouil Krasanakis
 */
public class CategoricalCrossEntropy extends Loss {
	private double epsilon;
	
	public CategoricalCrossEntropy() {
		this(1.E-12);
	}
	
	public CategoricalCrossEntropy(double epsilon) {
		this.epsilon = epsilon;
	}
	
	@Override
	public double evaluate(Tensor output, Tensor desired) {
		return -output.add(epsilon).selfLog().selfMultiply(desired).sum() / output.cast(Matrix.class).getRows();
	}
	
	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		return desired.multiply(output.add(epsilon).selfInverse()).selfMultiply(-1. / output.cast(Matrix.class).getRows());
	}
}
