package mklab.JGNN.examples;

import java.util.Arrays;

import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;

public class SimpleNetwork {

	public static void main(String[] args) throws Exception {
		int dims = 9;
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w1", new DenseMatrix(dims,1).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(2./dims)))
				.param("w2", new DenseMatrix(dims,dims).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(1./dims)))
				.param("w3", new DenseMatrix(1,dims).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(2./dims)))
				.param("b1", new DenseMatrix(dims,1))
				.param("b2", new DenseMatrix(dims,1))
				.operation("h1 = relu(w1@x+b1)")
				.operation("h2 = relu(w2@h1+b2)")
				.operation("yhat = w3@h2")
				.out("yhat")
				.print();
		BatchOptimizer optimizer = new BatchOptimizer(new Adam(0.01));
		// when no output is passed to training, the output is considered to be an error
		for(int i=0;i<100;i++) {
			for(int j=0;j<10;j++)
				modelBuilder.getModel().trainSampleDifference(optimizer, 
						Arrays.asList(new DenseMatrix(1,1).put(0,0,j/10.0)), 
						Arrays.asList(Tensor.fromDouble(j*j/100.)));
			optimizer.updateAll();
		}
		System.out.println(modelBuilder.runModel(new DenseMatrix(1,1).put(0,0,9/10.)).get("yhat").getPrediction().multiply(100));
		System.out.println(modelBuilder.get("b1").getPrediction());
		//run the wrapped model and obtain an internal variable prediction
	}
}
