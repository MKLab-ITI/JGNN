package mklab.JGNN.nn.pooling;

import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;

public class SoftMax extends NNOperation {
	private boolean colMode;
	public SoftMax() {
		this(false);
	}
	public SoftMax(boolean colMode) {
		super();
		this.colMode = colMode;
	}
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		/*if(colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = inputs.get(0).cast(Matrix.class);
			Matrix ret = matrix.zeroCopy().cast(Matrix.class);
			for(long row=0;row<ret.getRows();row++) {
				double max = Double.NEGATIVE_INFINITY;
				for(long col=0;col<matrix.getCols();col++) 
					max = Math.max(max, matrix.get(row, col));
				double sum = 0;
				for(long col=0;col<matrix.getCols();col++) {
					double element = Math.exp(matrix.get(row, col)-max);
					ret.put(row, col, element);
					sum += element;
				}
				if(sum!=0)
					for(long col=0;col<ret.getCols();col++)
						ret.put(row, col, ret.get(row, col)/sum);
			}
			return ret;
		}
		else if(!colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = inputs.get(0).cast(Matrix.class);
			Matrix ret = (Matrix)inputs.get(0).zeroCopy();
			for(long col=0;col<ret.getCols();col++) {
				double max = Double.NEGATIVE_INFINITY;
				for(long row=0;row<ret.getRows();row++)
					max = Math.max(max, matrix.get(row, col));
				double sum = 0;
				for(long row=0;row<ret.getRows();row++) {
					double element = Math.exp(matrix.get(row, col)-max);
					ret.put(row, col, element);
					sum += element;
				}
				if(sum!=0)
					for(long row=0;row<ret.getRows();row++)
						ret.put(row, col, ret.get(row, col)/sum);
			}
			return ret;
		}
		else {
			Tensor ret = inputs.get(0).zeroCopy();
			double max = Double.NEGATIVE_INFINITY;
			for(long i=0;i<ret.size();i++) 
				max = Math.max(max, ret.get(i));
			double sum = 0;
			for(long i=0;i<ret.size();i++) {
				double element = Math.exp(inputs.get(0).get(i)-max);
				ret.put(i, element);
				sum += element;
			}
			return ret.selfMultiply(1./sum);
		}*/
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		if(colMode && inputs.get(0) instanceof Matrix) {
			Matrix ret = (Matrix)inputs.get(0).zeroCopy();
			for(long i=0;i<ret.size();i++) 
				ret.put(i, Math.exp(inputs.get(0).get(i)));
			for(long row=0;row<ret.getRows();row++) {
				double sum = 0;
				for(long col=0;col<ret.getCols();col++)
					sum += ret.get(row, col);
				if(sum!=0)
					for(long col=0;col<ret.getCols();col++)
						ret.put(row, col, ret.get(row, col)/sum);
			}
			return ret;
		}
		else if(!colMode && inputs.get(0) instanceof Matrix) {
			Matrix ret = (Matrix)inputs.get(0).zeroCopy();
			for(long i=0;i<ret.size();i++) 
				ret.put(i, Math.exp(inputs.get(0).get(i)));
			for(long col=0;col<ret.getCols();col++) {
				double sum = 0;
				for(long row=0;row<ret.getRows();row++)
					sum += ret.get(row, col);
				if(sum!=0)
					for(long row=0;row<ret.getRows();row++)
						ret.put(row, col, ret.get(row, col)/sum);
			}
			return ret;
		}
		else {
			Tensor ret = inputs.get(0).zeroCopy();
			double sum = 0;
			for(long i=0;i<ret.size();i++) {
				ret.put(i, Math.exp(inputs.get(0).get(i)));
				sum += Math.exp(inputs.get(0).get(i));
			}
			return ret.selfMultiply(1./sum);
		}
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		if(colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) output;
			Matrix errorMatrix = (Matrix) error;
			Matrix ret = (Matrix) matrix.zeroCopy();
			for(long row=0;row<ret.getRows();row++) {
				double rowSum = 0;
				for(long col=0;col<ret.getCols();col++) 
					rowSum += matrix.get(row, col)*errorMatrix.get(row, col);
				for(long col=0;col<ret.getCols();col++) {
					double val = matrix.get(row, col);
					ret.put(row, col, (val*(1-val)*errorMatrix.get(row, col)-(rowSum-val*errorMatrix.get(row, col))*val));
				}
			}
			return ret;
		}
		else if(!colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) output;
			Matrix errorMatrix = (Matrix) error;
			Matrix ret = (Matrix) matrix.zeroCopy();
			for(long col=0;col<ret.getCols();col++) {
				double colSum = 0;
				for(long row=0;row<ret.getRows();row++)
					colSum += matrix.get(row, col)*errorMatrix.get(row, col);
				for(long row=0;row<ret.getRows();row++) {
					double val = matrix.get(row, col);
					ret.put(row, col, (val*(1-val)*errorMatrix.get(row, col)-(colSum-val*errorMatrix.get(row, col))*val));
				}
			}
			return ret;
		}
		else {
			throw new RuntimeException("Not implemented yet softmax for non-matrix inputs");
		}
	}
}