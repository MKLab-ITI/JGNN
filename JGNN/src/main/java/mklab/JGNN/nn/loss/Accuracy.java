package mklab.JGNN.nn.loss;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.Loss;

/**
 * Implements an accuracy {@link Loss} of row-by-row comparisons. Each row of
 * the output should have the same {@link Tensor#argmax()} value as the
 * corresponding row of desired values. This comparison has no derivative.
 * 
 * @author Emmanouil Krasanakis
 */
public class Accuracy extends Loss {
	/**
	 * Instantiates a row-by-row {@link Accuracy} loss. For this loss, each row of
	 * the output should have the same {@link Tensor#argmax()} value as the
	 * corresponding row of desired values. This comparison has no derivative.
	 */
	public Accuracy() {
	}

	@Override
	public double evaluate(Tensor output, Tensor desired) {
		Matrix moutput = output.cast(Matrix.class);
		Matrix mdesired = desired.cast(Matrix.class);
		double acc = 0;
		for (long row = 0; row < moutput.getRows(); row++)
			if (moutput.accessRow(row).argmax() == mdesired.accessRow(row).argmax())
				acc += 1;
		return -acc / moutput.getRows();
	}

	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		throw new RuntimeException("Not implemented");
	}
}
