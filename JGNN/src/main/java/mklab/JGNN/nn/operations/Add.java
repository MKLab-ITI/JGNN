package mklab.JGNN.nn.operations;

import java.util.ArrayList;
import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.ColumnRepetition;
import mklab.JGNN.core.matrix.RowRepetition;
import mklab.JGNN.nn.pooling.Sum;

/**
 * Implements a {@link NNOperation} that adds its two inputs.
 * 
 * @author Emmanouil Krasanakis
 */
public class Add extends NNOperation {
	@Override
	protected Tensor forward(List<Tensor> inputs) {
		if (inputs.size() != 2)
			throw new IllegalArgumentException();
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if (input0.size() == 1)
			return input1.add(input0.toDouble());
		if (input1.size() == 1)
			return input0.add(input1.toDouble());
		if (input0 instanceof Matrix && !(input1 instanceof Matrix))
			input1 = ((Matrix) input0).getCols() != input1.size()
					? new RowRepetition(input1, ((Matrix) input0).getCols())
					: new ColumnRepetition(((Matrix) input0).getRows(), input1);
		return input0.add(input1);
	}

	@Override
	protected Tensor partial(int inputId, List<Tensor> inputs, Tensor output, Tensor error) {
		Tensor input0 = inputs.get(0);
		Tensor input1 = inputs.get(1);
		if ((input0.size() == 1 && inputId == 0) || (input1.size() == 1 && inputId == 1)) {
			double val = 0;
			for (long pos : error.getNonZeroElements())
				val += error.get(pos);
			return Tensor.fromDouble(val);
		}
		if (inputId == 1 && input0 instanceof Matrix && !(input1 instanceof Matrix))
			return new Sum(((Matrix) input0).getCols() == input1.size()).run(error);
		if (inputId == 0 && input1 instanceof Matrix && !(input0 instanceof Matrix))
			return new Sum(((Matrix) input1).getRows() == input0.size()).run(error);
		return error;
	}

	@Override
	protected void autosize(ArrayList<Tensor> lastInputs) {
		Tensor input0 = lastInputs.get(0);
		Tensor input1 = lastInputs.get(1);
		if (input0 instanceof Matrix && input1 instanceof Matrix) {
			Matrix left = input0.cast(Matrix.class);
			Matrix right = input1.cast(Matrix.class);
			if (getInputs().get(0) instanceof Parameter && left.getRows() == 0 && left.getRowName().equals("?")) {
				if (right.getRows() == 0 && right.getRowName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(0))
						.set(left.zeroCopy(right.getRows(), left.getCols()).setRowName(right.getRowName())
								.setColName(left.getColName()).setDimensionName(left.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(0).describe());
			}
			if (getInputs().get(0) instanceof Parameter && left.getCols() == 0 && left.getColName().equals("?")) {
				if (right.getCols() == 0 && right.getColName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(0))
						.set(left.zeroCopy(left.getRows(), right.getCols()).setRowName(left.getRowName())
								.setColName(right.getColName()).setDimensionName(left.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(0).describe());
			}
			if (getInputs().get(0) instanceof Parameter && left.getRows() == 0 && left.getRowName().equals("?")) {
				if (right.getRows() == 0 && right.getRowName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(0))
						.set(left.zeroCopy(right.getRows(), left.getCols()).setRowName(right.getRowName())
								.setColName(left.getColName()).setDimensionName(left.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(0).describe());
			}
			if (getInputs().get(1) instanceof Parameter && right.getCols() == 0 && right.getColName().equals("?")) {
				if (left.getCols() == 0 && left.getColName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(1))
						.set(right.zeroCopy(right.getRows(), left.getCols()).setRowName(right.getRowName())
								.setColName(left.getColName()).setDimensionName(right.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(1).describe());
			}
			if (getInputs().get(1) instanceof Parameter && right.getRows() == 0 && right.getRowName().equals("?")) {
				if (left.getRows() == 0 && left.getRowName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(1))
						.set(right.zeroCopy(left.getRows(), right.getCols()).setRowName(left.getRowName())
								.setColName(right.getColName()).setDimensionName(right.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(1).describe());
			}
		} else if (input0 instanceof Matrix && !(input1 instanceof Matrix)) {
			Matrix matrix = input0.cast(Matrix.class);
			if (getInputs().get(0) instanceof Parameter && matrix.getCols() == 0 && matrix.getColName().equals("?")) {
				if (input1.size() == 0 && input1.getDimensionName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(0))
						.set(matrix.zeroCopy(matrix.getRows(), input1.size()).setRowName(matrix.getRowName())
								.setColName(input1.getDimensionName()).setDimensionName(matrix.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(0).describe());
			}
			if (getInputs().get(1) instanceof Parameter && input1.size() == 0
					&& input1.getDimensionName().equals("?")) {
				if (matrix.getCols() == 0 && matrix.getColName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(1))
						.set(input1.zeroCopy(matrix.getCols()).setDimensionName(matrix.getColName())).runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(1).describe());
			}
		} else {
			if (getInputs().get(0) instanceof Parameter && input0.size() == 0
					&& input0.getDimensionName().equals("?")) {
				if (input1.size() == 0 && input1.getDimensionName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(0))
						.set(input0.zeroCopy(input1.size()).setDimensionName(input1.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(0).describe());
			}
			if (getInputs().get(1) instanceof Parameter && input1.size() == 0
					&& input1.getDimensionName().equals("?")) {
				if (input0.size() == 0 && input0.getDimensionName().equals("?"))
					throw new RuntimeException("Cannot autosize based on two unknown dimensions");
				((Parameter) getInputs().get(1))
						.set(input1.zeroCopy(input0.size()).setDimensionName(input0.getDimensionName()))
						.runPrediction();
				if (debugging)
					System.out.println("Automatically sized parameter: " + getInputs().get(1).describe());

			}
		}
	}
}