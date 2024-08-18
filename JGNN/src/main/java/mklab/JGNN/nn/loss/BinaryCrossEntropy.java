package mklab.JGNN.nn.loss;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.Loss;

/**
 * Implements a binary cross-entropy {@link Loss}.<br>
 * For more than one output dimensions use {@link CategoricalCrossEntropy}
 * @author Emmanouil Krasanakis
 */
public class BinaryCrossEntropy extends Loss {
	private double epsilon;
	
	/**
	 * Initializes binary cross entropy with 1.E-12 epsilon value.
	 * For more than one output dimensions use {@link CategoricalCrossEntropy#CategoricalCrossEntropy()}
	 * @see #BinaryCrossEntropy(double)
	 */
	public BinaryCrossEntropy() {
		this(1.E-12);
	}
	/**
	 * Initializes binary cross entropy with and epsilon value 
	 * to bound its outputs in the range [log(epsilon), -log(epsilon)] instead of (-inf, inf).
	 * For more than one output dimensions use {@link CategoricalCrossEntropy#CategoricalCrossEntropy(double)}
	 * @param epsilon A very small positive <code>double</code>.
	 * @see #BinaryCrossEntropy()
	 */
	public BinaryCrossEntropy(double epsilon) {
		this.epsilon = epsilon;
	}
	
	@Override
	public double evaluate(Tensor output, Tensor desired) {
		Tensor outputComplement = output.multiply(-1).selfAdd(1);
		Tensor desiredComplement = output.multiply(-1).selfAdd(1);
		double ret =
				-output.add(epsilon).selfLog().selfMultiply(desired).sum()
			   -outputComplement.selfAdd(epsilon).selfLog().selfMultiply(desiredComplement).sum();
		return ret / output.cast(Matrix.class).getRows();
	}
	
	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		Tensor outputComplement = output.negative().selfAdd(1);
		Tensor desiredComplement = output.negative().selfAdd(1);
		return desired.multiply(output.add(epsilon).selfInverse())
			.selfAdd(
					desiredComplement.selfMultiply(outputComplement.add(epsilon).selfInverse())
					.selfNegative()).selfMultiply(-1./output.cast(Matrix.class).getRows());
	}
}
