package mklab.JGNN.nn.operations;

import java.util.Arrays;
import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.ColumnRepetition;
import mklab.JGNN.nn.pooling.Sum;

/**
 * Implements a {@link NNOperation} that multiplies its two inputs element-by-element.
 * 
 * @author Emmanouil Krasanakis
 */
public class Multiply extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if(input0.size()==1)
			return input1.multiply(input0.toDouble());
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
		if(input0.size()==1) {
			if(inputId==0) {
				return Tensor.fromDouble(error.dot(input1));
			}
			if(inputId==1 && input0.toDouble()!=0) 
				return error.multiply(input0.toDouble());
			if(inputId==1)
				return null;//this is the case where input0 is zero
		}
		if(inputId==0) {
			Tensor partialProduct = error.copy();
			if(input0 instanceof Matrix && !(input1 instanceof Matrix)) 
				input1 = new ColumnRepetition(((Matrix)input0).getRows(), input1);
			partialProduct.selfMultiply(input1);
			return partialProduct;
		}
		else if(inputId==1) {
			if(input0 instanceof Matrix && !(input1 instanceof Matrix)) 
				return (new Sum(true).forward(Arrays.asList(error.multiply(input0))));
			else {
				Tensor partialProduct = error.copy();
				partialProduct.selfMultiply(input0);
				return partialProduct;
			}
		}
		else
			throw new RuntimeException("Multiply takes exactly 2 arguments");
	}
	@Override
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		return outputNonLinearity * inputMass;
	}
}
