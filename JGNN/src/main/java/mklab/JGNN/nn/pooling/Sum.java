package mklab.JGNN.nn.pooling;

import java.util.List;

import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;

/**
 * Implements a {@link NNOperation} that performs row-wise or column-wise
 * sum reduction on vector tensors or matrices.
 * 
 * @author Emmanouil Krasanakis
 */
public class Sum extends NNOperation {
	private boolean colMode;
	public Sum() {
		this(false);
	}
	public Sum(boolean colMode) {
		super();
		this.colMode = colMode;
	}
	
	@Override
	public Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		if(colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) inputs.get(0);
			Tensor ret = new DenseTensor(matrix.getCols());
			for(Entry<Long, Long> entry : matrix.getNonZeroEntries()) {
				long row = entry.getKey();
				long col = entry.getValue();
				ret.put(col, ret.get(col) + matrix.get(row, col));
			}
			return ret;
		}
		else if(!colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) inputs.get(0);
			Tensor ret = new DenseTensor(matrix.getRows());
			for(Entry<Long, Long> entry : matrix.getNonZeroEntries()) {
				long row = entry.getKey();
				long col = entry.getValue();
				ret.put(row, ret.get(row) + matrix.get(row, col));
			}
			return ret;
		}
		else {
			double sum = 0;
			for(long i : inputs.get(0).getNonZeroElements())
				sum += inputs.get(0).get(i);
			return Tensor.fromDouble(sum);
		}
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		if(colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) inputs.get(0);
			Matrix ret = (Matrix) matrix.zeroCopy();
			for(Entry<Long, Long> entry : matrix.getNonZeroEntries()) {
				long row = entry.getKey();
				long col = entry.getValue();
				ret.put(col, col, ret.get(row, col)+error.get(col));
			}
			return ret;
		}
		else if(!colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) inputs.get(0);
			Matrix ret = (Matrix) matrix.zeroCopy();
			for(Entry<Long, Long> entry : matrix.getNonZeroEntries()) {
				long row = entry.getKey();
				long col = entry.getValue();
				ret.put(row, col, ret.get(row, col)+error.get(row));
			}
			return ret;
		}
		else {
			double errorValue = error.toDouble();
			return inputs.get(0).zeroCopy().setToOnes().multiply(errorValue);
		}
	}
}