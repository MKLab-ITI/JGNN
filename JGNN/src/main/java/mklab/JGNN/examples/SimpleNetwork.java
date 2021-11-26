package mklab.JGNN.examples;

import java.util.Arrays;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;
import mklab.JGNN.nn.optimizers.Regularization;

public class SimpleNetwork {

	public static void main(String[] args) throws Exception {
		int dims = 2;
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w1", new DenseMatrix(dims,1).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(2./dims)))
				.param("w2", new DenseMatrix(1,dims).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(2./dims)))
				.param("b1", new DenseMatrix(dims,1))
				.param("b2", new DenseMatrix(1,1))
				.operation("h1 = w1@x+b1")
				.operation("yhat = w2@h1+b2")
				.out("yhat")
				.print();
		Tensor x1 = new DenseTensor(10).setToRandom();
		Tensor y = x1.multiply(2);
		
		BatchOptimizer optimizer = new BatchOptimizer(new Regularization(new Adam(0.01), -0.00));
		// when no output is passed to training, the output is considered to be an error
		for(int i=0;i<10000;i++) {
			for(int j=0;j<10;j++)
				modelBuilder.getModel().trainL2(optimizer, 
						Arrays.asList(Matrix.fromDouble(x1.get(j))), 
						Arrays.asList(Matrix.fromDouble(y.get(j))));
			optimizer.updateAll();
			System.out.println(modelBuilder.runModel(Matrix.fromDouble(0.5)).get("yhat").getPrediction());
		}
		//run the wrapped model and obtain an internal variable prediction
	}
}
