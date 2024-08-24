package mklab.JGNN.nn.operations;

import java.util.List;
import java.util.Map.Entry;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that creates a version of adjacency matrices
 * with column-wise attention involving neighbor similarity.
 * 
 * @author Emmanouil Krasanakis
 */
public class Attention extends NNOperation {
	public Attention() {
	}

	@Override
	public Tensor forward(List<Tensor> inputs) {
		if (inputs.size() != 2)
			throw new IllegalArgumentException();
		Matrix adjacency = inputs.get(0).cast(Matrix.class);
		Matrix features = inputs.get(1).cast(Matrix.class);
		Matrix ret = adjacency.zeroCopy();
		for (Entry<Long, Long> pos : adjacency.getNonZeroEntries()) {
			long row = pos.getKey();
			long col = pos.getValue();
			if (row != col)
				ret.put(row, col, adjacency.get(row, col) * features.accessRow(row).dot(features.accessRow(col)));
		}
		return ret;
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Matrix features = inputs.get(1).cast(Matrix.class);
		Matrix errorMatrix = error.cast(Matrix.class);
		if (inputId == 0) {
			Tensor ret = inputs.get(0).zeroCopy();
			Matrix adjacency = inputs.get(0).cast(Matrix.class);
			for (long pos : output.getNonZeroElements())
				if (adjacency.get(pos) != 0)
					ret.put(pos, error.get(pos) * output.get(pos) / adjacency.get(pos));
			throw new RuntimeException("Should not create non-constant adjacency matrices");
		}
		Matrix ret = features.zeroCopy().cast(Matrix.class);
		for (Entry<Long, Long> pos : errorMatrix.getNonZeroEntries()) {
			long row = pos.getKey();
			long col = pos.getValue();
			if (row == col)
				continue;
			double err = errorMatrix.get(row, col);
			for (long i = 0; i < features.getCols(); i++)
				ret.accessRow(row).putAdd(i, features.get(col, i) * err);
		}
		return ret;
	}
}