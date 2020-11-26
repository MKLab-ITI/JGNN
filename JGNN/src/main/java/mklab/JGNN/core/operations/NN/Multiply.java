package mklab.JGNN.core.operations.NN;

import java.util.Arrays;
import java.util.List;

import mklab.JGNN.core.operations.NNOperation;
import mklab.JGNN.core.primitives.Matrix;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.primitives.matrix.ColumnRepetition;

public class Multiply extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if(input0 instanceof Matrix && !(input1 instanceof Matrix)) 
			input1 = new ColumnRepetition(((Matrix)input0).getRows(), input1);
		Tensor product = input0.copy();
		product.selfMultiply(input1);
		return product;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if(inputId==0) {
			Tensor partialProduct = error.copy();
			if(input0 instanceof Matrix && !(input1 instanceof Matrix)) 
				input1 = new ColumnRepetition(((Matrix)input0).getRows(), input1);
			partialProduct.selfMultiply(input1);
			return partialProduct;
		}
		else if(inputId==1) {
			if(input0 instanceof Matrix && !(input1 instanceof Matrix)) 
				return (new SumT().forward(Arrays.asList(error.multiply(input0))));
			else {
				Tensor partialProduct = error.copy();
				partialProduct.selfMultiply(input0);
				return partialProduct;
			}
		}
		else
			throw new RuntimeException("Multiply take exactly 2 arguments");
	}
}
