package mklab.JGNN.core.loss;
import mklab.JGNN.core.Loss;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Implements an accuracy {@link Loss} of row-by-row comparison.
 * @author Emmanouil Krasanakis
 */
public class Accuracy extends Loss {
	public Accuracy() {
	}
	
	@Override
	public double evaluate(Tensor output, Tensor desired) {
		Matrix moutput = output.cast(Matrix.class);
		Matrix mdesired = desired.cast(Matrix.class);
		double acc = 0;
		for(long row=0;row<moutput.getRows();row++)
			if(moutput.accessRow(row).argmax()==mdesired.accessRow(row).argmax())
				acc += 1;
		return -acc / moutput.getRows();
	}
	
	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		throw new RuntimeException("Not implemented");
	}
}
