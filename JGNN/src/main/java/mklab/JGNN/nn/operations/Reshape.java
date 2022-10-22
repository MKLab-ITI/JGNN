package mklab.JGNN.nn.operations;

import java.util.List;

import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that reshapes a matrix.
 * 
 * @author Emmanouil Krasanakis
 */
public class Reshape extends NNOperation {
	private long rows;
	private long cols;
	private String rowName = null;
	private String colName = null;
	
	public Reshape(long rows, long cols) {
		this.rows = rows;
		this.cols = cols;
		if(rows!=1 && cols!=1)
			throw new IllegalArgumentException("For the time being, reshape should have at least one of its dimensions be 1");
	}

	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if(inputs.size()!=1)
			throw new IllegalArgumentException();
		Tensor H = inputs.get(0);
		Matrix ret = rows==1?H.asRow():H.asColumn();
		ret.assertSize(rows*cols);
		return ret.setDimensionName(rowName, colName);
	}
	
	@Override
	public String getSimpleDescription() {
		return super.getSimpleDescription()+" ("+(rowName==null?"":(rowName+" "))+rows+","+(colName==null?"":(" "+colName+" "))+cols+")";
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor ret = inputs.get(0).zeroCopy();  // ensures typecast back to the correct matrix dims
		error.assertMatching(output);
		for(long i : error.getNonZeroElements())  // manual implementation of self-add to ignore all checks
			ret.put(i, error.get(i));
		return ret;
	}
	
	@Override
	public boolean isCachable() {
		return false;
	}

	public Reshape setDimensionName(String rowName, String colName) {
		this.rowName = rowName;
		this.colName = colName;
		return this;
	}
	
}