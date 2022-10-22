package mklab.JGNN.nn.operations;

import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;

/**
 * Implements a {@link NNOperation} that performs the equivalent of TensorFlow's gather operation.
 * 
 * @author Emmanouil Krasanakis
 */
public class Gather extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		Tensor index = inputs.get(0);
		Matrix H = (Matrix) inputs.get(1);
		Matrix ret = H.zeroCopy(index.size(), H.getCols()).setRowName(index.getDimensionName());
		for(int i=0;i<index.size();i++) {
			int pos = (int)index.get(i);
			for(int j=0;j<H.getCols();j++)
				ret.put(i, j, ret.get(i, j) + H.get(pos, j));
		}
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		if(inputId==0)
			return null;
		Tensor index = inputs.get(0);
		Matrix H = inputs.get(1).cast(Matrix.class);
		Matrix errorMatrix = error.cast(Matrix.class);
		Matrix derivative = H.zeroCopy().cast(Matrix.class);
		for(int i=0;i<index.size();i++) {
			int pos = (int)index.get(i);
			for(int j=0;j<H.getCols();j++) 
				derivative.put(pos, j, derivative.get(pos, j) + errorMatrix.get(i, j));
		}
		return derivative;
	}
}