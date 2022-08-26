package mklab.JGNN.core.loss;

import mklab.JGNN.core.Loss;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Implements a binary cross-entropy {@link Loss}.<br>
 * For more than one output dimensions use {@link CategoricalCrossEntropy}
 * @author Emmanouil Krasanakis
 */
public class BinaryCrossEntropy extends Loss {
	private double epsilon;
	
	public BinaryCrossEntropy() {
		this(1.E-12);
	}
	
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
