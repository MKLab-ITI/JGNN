package mklab.JGNN;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.nn.inputs.Constant;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.inputs.Variable;
import mklab.JGNN.nn.operations.Add;
import mklab.JGNN.nn.operations.Gather;
import mklab.JGNN.nn.operations.MatMul;
import mklab.JGNN.nn.operations.Multiply;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.pooling.Sum;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

public class NNOperationTest {
	@Test
	public void shouldTrainTowardsDirectObjective() {
		Optimizer optimizer = new Adam(1);
		
		Variable x = new Variable();
		NNOperation adder = (new Add())
				.addInput(x)
				.addInput(new Parameter(Tensor.fromDouble(0)));
		Model model = (new Model())
				.addInput(x)
				.addOutput(adder);
		for(int epoch=0;epoch<100;epoch++) {
			model.trainSampleDifference(optimizer, Arrays.asList(Tensor.fromDouble(1)), Arrays.asList(Tensor.fromDouble(2)));
			model.trainSampleDifference(optimizer, Arrays.asList(Tensor.fromDouble(2)), Arrays.asList(Tensor.fromDouble(3)));
			model.trainSampleDifference(optimizer, Arrays.asList(Tensor.fromDouble(3)), Arrays.asList(Tensor.fromDouble(4)));
		}
		Assert.assertEquals(7., model.predict(Arrays.asList(Tensor.fromDouble(6))).get(0).toDouble(), 0.1);
	}
	
	@Test
	public void shouldTrainTowardsLoss() {
		Optimizer optimizer = new Adam(1);
		
		Variable x = new Variable();
		NNOperation c = new Parameter(Tensor.fromDouble(0));
		NNOperation adder = (new Add())
				.addInput(x)
				.addInput(c);
		Model model = (new Model())
				.addInput(x)
				.addOutput(adder);
		
		Variable desiredOutput = new Variable();
		NNOperation inverse = new Multiply().addInput(desiredOutput).addInput(new Constant(Tensor.fromDouble(-1)));
		NNOperation diff = new Add()
				.addInput(inverse)
				.addInput(adder);
		NNOperation sqr = new Multiply().addInput(diff).addInput(diff);
		Model lossModel = (new Model())
				.addInput(x)
				.addInput(desiredOutput)
				.addOutput(sqr);
				
		for(int epoch=0;epoch<100;epoch++) {
			lossModel.trainSample(optimizer, Arrays.asList(Tensor.fromDouble(1), Tensor.fromDouble(2)));
			lossModel.trainSample(optimizer, Arrays.asList(Tensor.fromDouble(2), Tensor.fromDouble(3)));
			lossModel.trainSample(optimizer, Arrays.asList(Tensor.fromDouble(3), Tensor.fromDouble(4)));
		}
		Assert.assertEquals(7., model.predict(Arrays.asList(Tensor.fromDouble(6))).get(0).toDouble(), 0.001);
	}
	
	@Test
	public void shouldCovolveGraph() {
		int n = 5;
		int dims = 3;
		Variable from = new Variable();
		Variable to = new Variable();
		Matrix W = new SparseMatrix(n, n);
		W.put(1, 1, 1);
		W.put(0, 1, 1);
		W.put(1, 2, 1);
		NNOperation constW = new Constant(W);
		NNOperation h0 = new Parameter(new SparseMatrix(n, dims).setToRandom());
		NNOperation h1 = new MatMul().addInput(constW).addInput(h0);
		NNOperation h2 = new MatMul().addInput(constW).addInput(h1);
		NNOperation fromH = new Gather().addInput(from).addInput(h2);
		NNOperation toH = new Gather().addInput(to).addInput(h2);
		NNOperation sim = new Sum().addInput(new Multiply().addInput(fromH).addInput(toH));
		Model model = new Model()
			.addInput(from)
			.addInput(to)
			.addOutput(sim);
		
		Assert.assertEquals(1, model.predict(Arrays.asList(Tensor.fromDouble(0), Tensor.fromDouble(1))).get(0).size());
	}
	
	@Test
	public void builderShouldCreateModel() {
		Matrix W = new SparseMatrix(5, 5);
		W.put(1, 1, 1);
		W.put(0, 1, 1);
		W.put(1, 2, 1);
		Model model = new ModelBuilder()
				.var("u")
				.var("v")
				.constant("W", W)
				.param("H0", new DenseMatrix(5, 3).setToOnes())
				.operation("H1 = W * H0")
				.operation("u1 = H1[u]")
				.operation("v1 = H1[v]")
				.operation("mult = sum(u1.v1)")
				.out("mult")
				.getModel();
		
		Assert.assertEquals(1, model.predict(Arrays.asList(Tensor.fromDouble(0), Tensor.fromDouble(1))).get(0).size());
		Assert.assertEquals(6., model.predict(Arrays.asList(Tensor.fromDouble(0), Tensor.fromDouble(1))).get(0).toDouble(), 1.E-12);
	}
}
