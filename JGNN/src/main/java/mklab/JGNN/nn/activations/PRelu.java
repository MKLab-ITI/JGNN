package mklab.JGNN.nn.activations;

import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.ColumnRepetition;

public class PRelu extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		Tensor x = inputs.get(0);
		Tensor param = inputs.get(1);
		if(param.size()==1)
			return x.multiply(param.toDouble());
		if(x instanceof Matrix && !(param instanceof Matrix)) 
			param = new ColumnRepetition(((Matrix)x).getRows(), param);
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements()) {
			double val = x.get(i);
			ret.put(i, val>0?val:(val*param.get(i)));
		}
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor x = inputs.get(0);
		Tensor param = inputs.get(1);
		Tensor ret = inputs.get(inputId).zeroCopy();
		if(inputId==0) {
			if(param.size()==1)
				param = new ColumnRepetition(x.size(), param);
			if(x instanceof Matrix && !(param instanceof Matrix)) 
				param = new ColumnRepetition(((Matrix)x).getRows(), param);
			for(long i : error.getNonZeroElements()) {
				double val = x.get(i);
				ret.put(i, val>=0?error.get(i):(error.get(i)*param.get(i)));
			}
		}
		else if(inputId==1) {
			if(x instanceof Matrix && !(param instanceof Matrix)) {
				Matrix matrix = (Matrix)x;
				for(Entry<Long,Long> entry : matrix.getNonZeroEntries()) {
					long row = entry.getKey();
					long col = entry.getValue();
					double val = matrix.get(row, col);
					if(val<0)
						ret.put(col, ret.get(col)*((Matrix)error).get(row, col));
				}
			}
			else {
				for(long i : x.getNonZeroElements()) {
					double val = x.get(i);
					if(val<0)
						ret.put(i, error.get(i)*val);
				}
			}
		}
		else
			throw new RuntimeException("prelu takes exactly 2 arguments");
		return ret;
	}
	
	@Override
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		return outputNonLinearity*Math.sqrt(2);//TODO: check for this function
	}
}