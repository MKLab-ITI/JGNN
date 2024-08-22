package tutorial;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.datasets.Citeseer;
import mklab.JGNN.adhoc.datasets.Cora;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.BinaryCrossEntropy;
import mklab.JGNN.nn.optimizers.Adam;


/**
 * This implementation covers code of the Learning tutorial.
 * 
 * @author Emmanouil Krasanakis
 */
public class Learning {
	public static void main(String[] args) {
		Dataset dataset = new Cora();
		Matrix labels = dataset.labels().setDimensionName("samples", "labels");
		Matrix features = dataset.features().setDimensionName("samples", "features");
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();

		Slice sampleIds = dataset.samples().getSlice().shuffle();
		ModelTraining trainer = new ModelTraining()
			.setOptimizer(new Adam(0.01))
			.setEpochs(10)
			.setPatience(100)
			.setNumBatches(1)
			.setVerbose(true)
			.setParallelizedStochasticGradientDescent(false)
			.setLoss(new BinaryCrossEntropy());
		
		ModelBuilder modelBuilder = new ModelBuilder()
				.config("features", numFeatures)
				.config("classes", numClasses)
				.config("regularize", 1.E-5)
				.var("x")
				.operation("h = relu(x@matrix(features, 16, regularize)+vector(16))")
				.operation("yhat = softmax(h@matrix(16, classes)+vector(classes), dim: 'row')")
				.out("yhat")
				.assertBackwardValidity();
		
		System.out.println(modelBuilder.getExecutionGraphDot());
		Model model = modelBuilder
				.getModel()
				.init(new XavierNormal())
				.train(trainer, features, labels, 
						sampleIds.range(0, 0.1), 
						sampleIds.range(0.2, 0.4));
		
		double acc = 0;
		for(Long node : sampleIds.range(0.75, 1)) {
			Matrix nodeFeatures = features.accessRow(node).asRow();
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/sampleIds.range(0.75, 1).size());
		
	}

}
