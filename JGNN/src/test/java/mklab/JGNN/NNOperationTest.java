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
	public void shouldPerformAddition() {
		Assert.assertEquals((new Add()).run(Tensor.fromDouble(1), Tensor.fromDouble(2)).toDouble(), 3, 0);
	}
	
	
	@Test
	public void shouldCreateOperationTree() {
		NNOperation add = new Add();
		add.addInput(new Constant(Tensor.fromDouble(1)));
		add.addInput(new Constant(Tensor.fromDouble(2)));
		Assert.assertEquals(add.getInputs().size(), 2, 0);
		Assert.assertEquals(add.getInputs().get(0).getInputs().size(), 0, 0);
		Assert.assertEquals(add.getInputs().get(0).getOutputs().size(), 1, 0);
		Assert.assertEquals(add.getInputs().get(0).getOutputs().get(0), add);
	}
	
	@Test
	public void shouldNotRememberSimpleRun() {
		NNOperation add = new Add();
		add.run(Tensor.fromDouble(1), Tensor.fromDouble(2));
		Assert.assertNull(add.getPrediction());
		Assert.assertNull(add.getLastTapeError());
	}

	@Test
	public void shouldRememberPrediction() {
		NNOperation add = new Add();
		add.addInput(new Constant(Tensor.fromDouble(1)));
		add.addInput(new Constant(Tensor.fromDouble(2)));
		add.runPrediction();
		Assert.assertEquals(add.getPrediction().toDouble(), 3, 0);
	}
	
	@Test
	public void shouldParseInputUpdates() {
		NNOperation add = new Add();
		Constant constant;
		add.addInput(new Constant(Tensor.fromDouble(1)));
		add.addInput(constant = new Constant(Tensor.fromDouble(2)));
		add.runPrediction();
		Assert.assertEquals(add.getPrediction().toDouble(), 3, 0);
		constant.setTo(Tensor.fromDouble(0));
		add.clearPrediction();
		add.runPrediction();
		Assert.assertEquals(add.getPrediction().toDouble(), 1, 0);
	}

	@Test
	public void shouldPerformMultiplication() {
		Assert.assertEquals((new Add()).run(Tensor.fromDouble(1), Tensor.fromDouble(2)).toDouble(), 3, 0);
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
}
