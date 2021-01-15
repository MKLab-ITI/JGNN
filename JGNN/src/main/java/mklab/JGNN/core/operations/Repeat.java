package mklab.JGNN.core.operations;

import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.ColumnRepetition;

public class Repeat extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=2)
			throw new IllegalArgumentException();
		int repetitions = (int)inputs.get(1).toDouble();
		return new ColumnRepetition(repetitions, inputs.get(0));
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		if(inputId==1)
			return null;
		Tensor ret = inputs.get(0).zeroCopy();
		Matrix errorMatrix = (Matrix)error;
		for(Entry<Long, Long> element : errorMatrix.getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			ret.put(col, ret.get(col) + errorMatrix.get(row, col));
		}
		return ret;
	}
	
}