package mklab.JGNN.examples;

import java.util.Arrays;

import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.inputs.Variable;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.optimizers.Adam;
import mklab.JGNN.core.tensor.DenseTensor;

public class ModelSimpleTraining {

	public static void main(String[] args) throws Exception {
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x") // first argument
				.var("y") // second argument
				.param("a", Tensor.fromDouble(1))
				.param("b", Tensor.fromDouble(0))
				.operation("yhat = a*x+b")
				.operation("error = (y-yhat)*(y-yhat)")
				.out("error")
				.print();
		Optimizer optimizer = new Adam(0.1);
		// when no output is passed to training, the output is considered to be an error
		for(int i=0;i<200;i++)
			modelBuilder.getModel().trainSample(optimizer, Arrays.asList(new DenseTensor(1,2,3,4,5), new DenseTensor(3, 5, 7, 9, 11)));
		//run the wrapped model and obtain an internal variable prediction
		System.out.println(modelBuilder.runModel(Tensor.fromDouble(2), Tensor.fromDouble(0)).get("yhat").getPrediction());
	}
}
