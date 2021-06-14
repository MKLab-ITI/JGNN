package mklab.JGNN.examples;

import java.util.Arrays;

import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.optimizers.Adam;
import mklab.JGNN.core.optimizers.BatchOptimizer;

public class SimpleNetwork {

	public static void main(String[] args) throws Exception {
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w1", new DenseMatrix(9,1).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(2./9)))
				.param("w2", new DenseMatrix(9,9).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(2./18)))
				.param("w3", new DenseMatrix(1,9).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(2./9)))
				.operation("h1 = relu(w1@x)")
				.operation("h2 = relu(w2@h1)")
				.operation("yhat = w3@h2")
				.out("yhat")
				.print();
		BatchOptimizer optimizer = new BatchOptimizer(new Adam(0.1));
		// when no output is passed to training, the output is considered to be an error
		for(int i=0;i<100;i++) {
			for(int j=0;j<10;j++)
				modelBuilder.getModel().trainSampleDifference(optimizer, Arrays.asList(new DenseMatrix(1,1).put(0,0,j)), Arrays.asList(Tensor.fromDouble(j*j)));
			optimizer.updateAll();
			System.out.println(modelBuilder.runModel(new DenseMatrix(1,1).put(0,0,9)).get("yhat").getPrediction());
		}
		//run the wrapped model and obtain an internal variable prediction
	}
}
