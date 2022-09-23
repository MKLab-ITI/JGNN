package primitives;

import java.util.Arrays;

import mklab.JGNN.nn.Optimizer;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.nn.optimizers.Adam;

/**
 * Demonstrates model builder internal node access that allows training with
 * a symbolically defined loss function.
 * 
 * @author Emmanouil Krasanakis
 */
public class ModelBuilderInternals {
	public static void main(String[] args) throws Exception {
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x") // first argument
				.var("y") // second argument
				.param("a", Tensor.fromDouble(1))
				.param("b", Tensor.fromDouble(0))
				.operation("yhat = a*x+b")
				.operation("error = mean((y-yhat)*(y-yhat))")
				.out("error")
				.print();
		Optimizer optimizer = new Adam(0.1);
		// when no output is passed to training, the output is considered to be an error
		for(int i=0;i<10000;i++) {
			modelBuilder.getModel().trainTowardsZero(optimizer, 
					Arrays.asList(new DenseTensor(1,2,3,4,5), 
							      new DenseTensor(3,5,7,9,11)));  // learn y = 2x+1
		}
		//run the wrapped model and obtain an internal variable prediction
		System.out.println(modelBuilder.runModel(Tensor.fromDouble(2.3), Tensor.fromDouble(0)).get("yhat").getPrediction());
	}
}
