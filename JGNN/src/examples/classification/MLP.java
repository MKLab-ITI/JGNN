package classification;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.datasets.Citeseer;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.Accuracy;
import mklab.JGNN.nn.loss.BinaryCrossEntropy;
import mklab.JGNN.nn.optimizers.Adam;

/**
 * Demonstrates classification with a two-layer perceptron
 * 
 * @author Emmanouil Krasanakis
 */
public class MLP {

	public static void main(String[] args) {
		Dataset dataset = new Citeseer();
		
		System.out.println("Nodes\t: "+dataset.samples().size());
		System.out.println("Labels\t: "+dataset.labels().describe());
		System.out.println("Features: "+dataset.features().describe());
		
		long numFeatures = dataset.features().getCols();
		long numClasses = dataset.labels().getCols();
		ModelBuilder modelBuilder = new ModelBuilder()
				.config("features", numFeatures)
				.config("classes", numClasses)
				.config("regularization", 0.005)
				.var("x")
				.operation("h = relu(x@matrix(features, classes, regularization)+vector(classes))")
				.operation("yhat = softmax(h@matrix(classes, classes)+vector(classes), row)")
				.out("yhat")
				.print();
		System.out.println(modelBuilder.getExecutionGraphDot());
		
		Slice nodeIds = dataset.samples().getSlice().shuffle(100);
		
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(3000)
				.setPatience(300)
				.setNumBatches(20)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(new BinaryCrossEntropy())
				.setVerbose(true)
				.setValidationLoss(new Accuracy())
				.train(new XavierNormal().apply(modelBuilder.getModel()), 
						dataset.features(), 
						dataset.labels(),
						nodeIds.range(0, 0.7), nodeIds.range(0.7, 0.8));
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : nodeIds.range(0.8, 1)) {
			Matrix nodeFeatures = dataset.features().accessRow(node).asRow();
			Matrix nodeLabels = dataset.labels().accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += output.argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/nodeIds.range(0.8, 1).size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
