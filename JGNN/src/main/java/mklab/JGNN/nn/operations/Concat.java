package mklab.JGNN.nn.operations;

import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.WrapCols;

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
		List<Tensor> cols0 = inputs.get(0).cast(Matrix.class).accessColumns();
		List<Tensor> cols1 = inputs.get(1).cast(Matrix.class).accessColumns();
		cols0.addAll(cols1);
		if(inputs.get(0).cast(Matrix.class).getRowName()!=inputs.get(1).cast(Matrix.class).getRowName())
			throw new RuntimeException("Cannot concatenate: "+inputs.get(0).describe()+" and "+inputs.get(1).describe());
		return new WrapCols(cols0).setDimensionName(inputs.get(0).cast(Matrix.class).getRowName(), null);
		/*Matrix matrix0 = (Matrix)inputs.get(0);
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
		return matrix;*/
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Matrix matrix0 = (Matrix)inputs.get(0);
		Matrix matrix1 = (Matrix)inputs.get(1);
		long start = inputId==0?0:matrix0.getCols();
		long end = inputId==0?matrix0.getCols():(matrix0.getCols()+matrix1.getCols());
		return error.cast(Matrix.class).accessColumns(Tensor.fromRange(start, end)).setDimensionName(null, null);
		/*Matrix inputError = (Matrix)inputs.get(inputId).zeroCopy();
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
		return inputError;*/
	}

}
