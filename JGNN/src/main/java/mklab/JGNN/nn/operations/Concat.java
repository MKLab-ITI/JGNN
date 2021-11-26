package mklab.JGNN.nn.operations;

import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;

/**
 * Implements a {@link NNOperation} that concatenates its two matrix inputs.
 * 
 * @author Emmanouil Krasanakis
 */
public class Concat extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		Matrix matrix0 = (Matrix)inputs.get(0);
		Matrix matrix1 = (Matrix)inputs.get(1);
		if(matrix0.getRows()!=matrix1.getRows())
			throw new IllegalArgumentException();
		Matrix matrix = (matrix0 instanceof SparseMatrix && matrix1 instanceof SparseMatrix)
				?new SparseMatrix(matrix0.getRows(), matrix0.getCols()+matrix1.getCols())
				:new DenseMatrix(matrix0.getRows(), matrix0.getCols()+matrix1.getCols());
		for(Entry<Long, Long> entry : matrix0.getNonZeroEntries())
			matrix.put(entry.getKey(), entry.getValue(), matrix0.get(entry.getKey(), entry.getValue()));
		for(Entry<Long, Long> entry : matrix1.getNonZeroEntries())
			matrix.put(entry.getKey(), matrix0.getCols()+entry.getValue(), matrix1.get(entry.getKey(), entry.getValue()));
		return matrix;
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Matrix inputError = (Matrix)inputs.get(inputId).zeroCopy();
		Matrix matrix0 = (Matrix)inputs.get(0);
		Matrix matrix1 = (Matrix)inputs.get(1);
		Matrix errorMatrix = (Matrix)error;
		long start = inputId==0?0:matrix0.getCols();
		long end = inputId==0?matrix0.getCols():(matrix0.getCols()+matrix1.getCols());
		for(Entry<Long, Long> entry : matrix0.getNonZeroEntries()) {
			long row = entry.getKey();
			long col = entry.getValue();
			if(col>=start && col<end)
				inputError.put(row, col-start, errorMatrix.get(row, col));
		}
		return inputError;
	}

}
