package mklab.JGNN.core.operations.NN;

import java.util.Arrays;
import java.util.List;

import mklab.JGNN.core.operations.NNOperation;
import mklab.JGNN.core.primitives.Matrix;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.primitives.matrix.ColumnRepetition;

public class Add extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if(input0 instanceof Matrix && !(input1 instanceof Matrix)) 
			input1 = new ColumnRepetition(((Matrix)input0).getRows(), input1);
		Tensor product = input0.copy();
		product.selfAdd(input1);
		return product;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if(inputId==1 && input0 instanceof Matrix && !(input1 instanceof Matrix)) 
			return new SumT().forward(Arrays.asList(error));
		return error;
	}
}