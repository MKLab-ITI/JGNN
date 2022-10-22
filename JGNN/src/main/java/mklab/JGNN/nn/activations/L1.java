package mklab.JGNN.nn.activations;

import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;


/**
 * Implements a {@link NNOperation} that performs a L1 transformation of its single input
 * by row or column.
 * 
 * @author Emmanouil Krasanakis
 */
public class L1 extends NNOperation {
	private boolean colMode;
	public L1() {
		this(false);
	}
	public L1(boolean colMode) {
		super();
		this.colMode = colMode;
	}
	
	@Override
	public Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		Tensor ret = inputs.get(0).copy();
		if(ret instanceof Matrix) {
			if(colMode) {
				Matrix matrix = ret.cast(Matrix.class);
				Tensor sums = new DenseTensor(matrix.getRows());
				for(Entry<Long, Long> pos : matrix.getNonZeroEntries()) {
					long row = pos.getKey();
					long col = pos.getValue();
					sums.putAdd(row, matrix.get(row, col));
				}
				for(Entry<Long, Long> pos : matrix.getNonZeroEntries()) {
					long row = pos.getKey();
					long col = pos.getValue();
					double div = sums.get(row);
					matrix.put(row, col, Math.abs(matrix.get(row, col))/div);
				}
			}
			else {
				Matrix matrix = ret.cast(Matrix.class);
				Tensor sums = new DenseTensor(matrix.getCols());
				for(Entry<Long, Long> pos : matrix.getNonZeroEntries()) {
					long row = pos.getKey();
					long col = pos.getValue();
					sums.putAdd(col, matrix.get(row, col));
				}
				for(Entry<Long, Long> pos : matrix.getNonZeroEntries()) {
					long row = pos.getKey();
					long col = pos.getValue();
					double div = sums.get(col);
					if(div!=0)
						matrix.put(row, col, Math.abs(matrix.get(row, col))/div);
				}
			}
		}
		else 
			return ret.setToProbability();
		return ret;
	}
	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor ret = error.zeroCopy();
		Tensor input = inputs.get(0);
		for(long pos : error.getNonZeroElements()) {
			double nom = input.get(pos);
			if(nom==0)
				continue;
			double denom = nom/output.get(pos);
			double sgn = nom>0?1:-1;
			ret.put(pos, sgn*(1.-nom/denom)/denom);
		}
		return ret;
	}
}