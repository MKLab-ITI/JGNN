package mklab.JGNN.nn.pooling;

import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;


/**
 * Implements a {@link NNOperation} that performs row-wise or column-wise
 * maximum reduction on vector tensors or matrices.
 * 
 * @author Emmanouil Krasanakis
 */
public class Max extends NNOperation {
	private boolean colMode;
	public Max() {
		this(false);
	}
	public Max(boolean colMode) {
		super();
		this.colMode = colMode;
	}
	
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		if(colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) inputs.get(0);
			Tensor ret = new DenseMatrix(1, matrix.getCols());
			for(Entry<Long, Long> entry : matrix.getNonZeroEntries()) {
				long row = entry.getKey();
				long col = entry.getValue();
				ret.put(col, Math.max(ret.get(col), matrix.get(row, col)));
			}
			return ret;
		}
		else if(!colMode && inputs.get(0) instanceof Matrix) {
			Matrix matrix = (Matrix) inputs.get(0);
			Tensor ret = new DenseMatrix(matrix.getRows(), 1);
			for(Entry<Long, Long> entry : matrix.getNonZeroEntries()) {
				long row = entry.getKey();
				long col = entry.getValue();
				ret.put(row, Math.max(ret.get(row), matrix.get(row, col)));
			}
			return ret;
		}
		else {
			double maxValue = 0;
			for(long i : inputs.get(0).getNonZeroElements())
				maxValue = Math.max(maxValue, inputs.get(0).get(i));
			return Tensor.fromDouble(maxValue);
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
				if(matrix.get(row, col) == output.get(col))
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
				if(matrix.get(row, col) == output.get(row))
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