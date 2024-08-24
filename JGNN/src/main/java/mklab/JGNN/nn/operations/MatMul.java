package mklab.JGNN.nn.operations;

import java.util.ArrayList;
import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.core.Tensor;

/**
 * Implements a {@link NNOperation} that multiplies its two matrix inputs.
 * 
 * @author Emmanouil Krasanakis
 */
public class MatMul extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if (inputs.size() != 2)
			throw new IllegalArgumentException();
		Matrix W = inputs.get(0).cast(Matrix.class);
		Matrix H = inputs.get(1).cast(Matrix.class);
		return W.matmul(H);
	}

	protected boolean isInputNeededForDerivative(int inputId) {
		return !getInputs().get(1 - inputId).isConstant();
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Matrix errorMatrix = (Matrix) error;
		if (inputId == 0) {
			Matrix H = inputs.get(1).cast(Matrix.class);
			errorMatrix = errorMatrix.matmul(H, false, true);
		} else if (inputId == 1) {
			Matrix W = inputs.get(0).cast(Matrix.class);
			errorMatrix = W.matmul(errorMatrix, true, false);
		}
		return errorMatrix;
	}

	@Override
	public double getNonLinearity(int inputId, double inputMass, double outputNonLinearity) {
		return outputNonLinearity * inputMass;
	}

	@Override
	protected void autosize(ArrayList<Tensor> lastInputs) {
		Matrix left = lastInputs.get(0).cast(Matrix.class);
		Matrix right = lastInputs.get(1).cast(Matrix.class);
		if (getInputs().get(0) instanceof Parameter && left.getCols() == 0 && left.getColName().equals("?")) {
			if (right.getRows() == 0 && right.getRowName().equals("?"))
				throw new RuntimeException("Cannot autosize based on two unknown dimensions");
			((Parameter) getInputs().get(0))
					.set(left.zeroCopy(left.getRows(), right.getRows()).setRowName(left.getRowName())
							.setColName(right.getRowName()).setDimensionName(left.getDimensionName()))
					.runPrediction();
			if (debugging)
				System.out.println("Automatically sized parameter: " + getInputs().get(0).describe());
		}
		if (getInputs().get(1) instanceof Parameter && right.getRows() == 0 && right.getRowName().equals("?")) {
			if (left.getCols() == 0 && left.getColName().equals("?"))
				throw new RuntimeException("Cannot autosize based on two unknown dimensions");
			((Parameter) getInputs().get(1))
					.set(right.zeroCopy(left.getCols(), right.getCols()).setRowName(left.getColName())
							.setColName(right.getColName()).setDimensionName(right.getDimensionName()))
					.runPrediction();
			if (debugging)
				System.out.println("Automatically sized parameter: " + getInputs().get(1).describe());
		}
	}
}
