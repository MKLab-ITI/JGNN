package mklab.JGNN;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

import mklab.JGNN.core.Distribution;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.distribution.Normal;
import mklab.JGNN.core.distribution.Uniform;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.inputs.Constant;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.inputs.Variable;
import mklab.JGNN.nn.operations.Add;
import mklab.JGNN.nn.operations.Multiply;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;
import mklab.JGNN.nn.optimizers.GradientDescent;
import mklab.JGNN.nn.optimizers.Regularization;

public class LearningTest {
	/*
	public Tensor classify(BatchOptimizer optimizer, int epochs) {
		Dataset dataset = new Datasets.Lymphography();
		Matrix labels = dataset.nodes().oneHot(dataset.labels()).asTransposed();
		Matrix features = dataset.nodes().oneHot(dataset.features()).asTransposed();
		
		long numFeatures = features.getRows();
		long numClasses = labels.getRows();
		int dims = 2;
		Distribution distribution = new Uniform(-1,1).setSeed(0);
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w1", new DenseMatrix(numClasses, numFeatures).setToRandom(distribution).selfMultiply(Math.sqrt(1./dims)))
				.operation("yhat = sigmoid(w1@x)")
				.out("yhat");
		
		double acc0 = 0;
		double accFinal = 0;
		Tensor output = null;
		for(int epoch=0;epoch<epochs;epoch++) {
			modelBuilder.getModel().trainSampleDifference(optimizer, 
					Arrays.asList(features), 
					Arrays.asList(labels))
					.get(0)
					.subtract(labels);
			
			double acc = 0;
			for(Long node : dataset.nodes().getIds()) {
				Matrix nodeFeatures = features.getCol(node).asColumn();
				Matrix nodeLabels = labels.getCol(node).asColumn();
				output = modelBuilder.getModel().predict(nodeFeatures).get(0);
				acc += (output.argmax()==nodeLabels.argmax()?1:0);
			}
			optimizer.updateAll();
			if(epoch==0)
				acc0 = acc/dataset.nodes().size();
			else 
				accFinal = acc/dataset.nodes().size();
		}
		Assert.assertTrue(acc0<accFinal);
		return output;
	}
	
	@Test
	public void shouldTrainDatasetTowardsCorrectClassificationWithGradients() {
		classify(new BatchOptimizer(new Regularization(new GradientDescent(0.1), 0.001)), 5);
	}

	@Test
	public void shouldTrainDatasetTowardsCorrectClassificationWithAdam() {
		classify(new BatchOptimizer(new Regularization(new Adam(), 0.001)), 5);
	}

	@Test
	public void shouldBeTrainableWithBatchOptimizerOfOneSample() {
		classify(new BatchOptimizer(new Regularization(new Adam(), 0.001), 1), 5);
	}
	
	@Test
	public void shouldTrainDatasetTowardsCorrectClassificationWithNDAdam() {
		classify(new BatchOptimizer(new Regularization(new Adam(true, 0.1), 0.001)), 5);
	}

	@Test
	public void shouldPerformOptimizerResetAndControlledExperiments() {
		BatchOptimizer batchOptimizer = new BatchOptimizer(new Regularization(new Adam(true, 0.1), 0.001));
		Tensor result1 = classify(batchOptimizer, 5);
		batchOptimizer.reset();
		Tensor result2 = classify(batchOptimizer, 5);
		Assert.assertEquals(result1.subtract(result2).norm(), 0, 0);
	}

	
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
				.operation("H1 = W @ H0")
				.operation("u1 = H1[u]")
				.operation("v1 = H1[v]")
				.operation("mult = sum(u1*v1)")
				.out("mult")
				.getModel();
		
		Assert.assertEquals(1, model.predict(Arrays.asList(Tensor.fromDouble(0), Tensor.fromDouble(1))).get(0).size());
		Assert.assertEquals(6., model.predict(Arrays.asList(Tensor.fromDouble(0), Tensor.fromDouble(1))).get(0).toDouble(), 1.E-12);
	}
	*/
}
