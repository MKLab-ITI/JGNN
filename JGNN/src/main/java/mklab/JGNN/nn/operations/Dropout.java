package mklab.JGNN.nn.operations;

import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.ColumnRepetition;

/**
 * Implements a {@link NNOperation} that converts its first argument to a {@link ColumnRepetition} matrix
 * with a number of columns equal to the second argument.
 * 
 * @author Emmanouil Krasanakis
 */
public class Dropout extends NNOperation {
	private boolean enabled = false;
	
	public boolean isEnabled() {
		return enabled;
	}
	
	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}
	
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		double value = inputs.get(1).toDouble();
		if(value<0 || value>1)
			throw new IllegalArgumentException();
		if(!enabled || value==0)
			return inputs.get(0);
		Tensor input = inputs.get(0);
		Tensor ret = inputs.get(0).zeroCopy();
		for(long pos : input.getNonZeroElements())
			if(Math.random()<value)
				ret.put(pos, input.get(pos)/value);
		return ret;
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		if(inputId==1)
			return null;
		if(!enabled)
			return error;
		double value = inputs.get(1).toDouble();
		Tensor ret = output.zeroCopy();
		for(long pos : output.getNonZeroElements())
			if(output.get(pos)!=0)
				ret.put(pos, error.get(pos)*value);
		return ret;
	}
	
	@Override
	public boolean isConstant() {
		return getInputs().get(0).isConstant();
	}
	
}