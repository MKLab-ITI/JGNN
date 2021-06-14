package mklab.JGNN.core.operations;

import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that multiplies its two matrix inputs.
 * 
 * @author Emmanouil Krasanakis
 */
public class MatMul extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		Matrix W = (Matrix) inputs.get(0);
		Matrix H = (Matrix) inputs.get(1);
		return W.matmul(H);
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Matrix errorMatrix = (Matrix)error;
		Matrix W = (Matrix) inputs.get(0);
		Matrix H = (Matrix) inputs.get(1);
		if(inputId==0)
			errorMatrix = errorMatrix.matmul(H, false, true);
		else if(inputId==1) 
			errorMatrix = W.matmul(errorMatrix, true, false);
		return errorMatrix;
	}
}
