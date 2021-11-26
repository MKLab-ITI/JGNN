package mklab.JGNN.nn.operations;

import java.util.Arrays;
import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.ColumnRepetition;
import mklab.JGNN.nn.pooling.SumT;

/**
 * Implements a {@link NNOperation} that adds its two inputs.
 * 
 * @author Emmanouil Krasanakis
 */
public class Add extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if(input0.size()==1)
			return input1.add(input0.toDouble());
		if(input1.size()==1)
			return input0.add(input1.toDouble());
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
		if((input0.size()==1 && inputId==0) || (input1.size()==1 && inputId==1)) {
			double val = 0;
			for(long pos : error.getNonZeroElements())
				val += error.get(pos);
			return Tensor.fromDouble(val);
		}
		if(inputId==1 && input0 instanceof Matrix && !(input1 instanceof Matrix)) 
			return new SumT().forward(Arrays.asList(error));
		return error;
	}
}