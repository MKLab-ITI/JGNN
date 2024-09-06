package classification;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.adhoc.datasets.Citeseer;
import mklab.JGNN.adhoc.train.SampleClassification;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.Accuracy;
import mklab.JGNN.nn.loss.BinaryCrossEntropy;
import mklab.JGNN.nn.loss.report.VerboseLoss;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.nn.optimizers.GradientDescent;
import mklab.JGNN.nn.optimizers.Adam;

/**
 * Demonstrates classification with logistic regression.
 * 
 * @author Emmanouil Krasanakis
 */
public class LogisticRegression {

	public static void main(String[] args) {
		Dataset dataset = new Citeseer();
		
		System.out.println("Nodes\t: "+dataset.samples().size());
		System.out.println("Labels\t: "+dataset.labels().describe());
		System.out.println("Features: "+dataset.features().describe());

		long numFeatures = dataset.features().getCols();
		long numClasses = dataset.labels().getCols();
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w", new DenseMatrix(numFeatures, numClasses)
						.setDimensionName("features", "labels")
						.setToRandom().selfAdd(-0.5).selfMultiply(1./Math.sqrt(numFeatures+numClasses)))
				.param("b", new DenseTensor(numClasses).setDimensionName("labels"))
				.operation("yhat = sigmoid(x@w)")
				.out("yhat")
				.print();
		
		Slice nodeIds = dataset.samples().getSlice().shuffle(100);
		
		
		long tic = System.currentTimeMillis();
		ModelTraining trainer = new SampleClassification()
				.setFeatures(dataset.features())
				.setOutputs(dataset.labels())
				.setTrainingSamples(nodeIds.range(0, 0.6))
				.setValidationSamples(nodeIds.range(0.6, 0.8))
				.setOptimizer(new GradientDescent(0.01))
				.setEpochs(600)
				.setOptimizer(new Adam(0.01))
				.setNumBatches(10)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(new BinaryCrossEntropy())
				.setValidationLoss(new VerboseLoss(new Accuracy()));
			
		Model model = modelBuilder.getModel()
				.init(new XavierNormal())
				.train(trainer);
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : nodeIds.range(0.8, 1)) {
			Matrix nodeFeatures = dataset.features().accessRow(node).asRow();
			Matrix nodeLabels = dataset.features().accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/nodeIds.range(0.8, 1).size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
