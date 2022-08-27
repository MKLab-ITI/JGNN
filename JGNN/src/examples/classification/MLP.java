package classification;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelBuilder;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.loss.BinaryCrossEntropy;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.GradientDescent;

/**
 * <b>Tasks</b>	       : node classification<br>
 * <b>Algorithms</b>   : multilayer perceptron<br>
 * <b>Datasets</b>     : Lymphography<br>
 * <b>Demonstrates</b> : dataset parsing, {@link ModelBuilder}, {@link ModelTraining}, accuracy evaluation, named dimensions, automatic parsing, {@link Initializer}<br>
 * 
 * @author Emmanouil Krasanakis
 */
public class MLP {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		IdConverter nodes = dataset.nodes();
		Matrix labels = nodes.setDimensionName("nodes", "labels").oneHot(dataset.getLabels());
		Matrix features = nodes.setDimensionName("nodes", "features").oneHot(dataset.getFeatures());
		
		System.out.println("Nodes\t: "+dataset.nodes().size());
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		ModelBuilder modelBuilder = new ModelBuilder()
				.config("features", numFeatures)
				.config("classes", numClasses)
				.var("x")
				.operation("h = relu(x@matrix(features, classes)+vector(classes))")
				.operation("yhat = softmax(h@matrix(classes, classes)+vector(classes), row)")
				.out("yhat")
				.print();
		System.out.println(modelBuilder.getExecutionGraphDot());
		
		Slice nodeIds = dataset.nodes().getIds().shuffle(100);
		
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new GradientDescent(0.01))
				.setEpochs(3000)
				.setPatience(300)
				.setNumBatches(10)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(new BinaryCrossEntropy())
				.train(new XavierNormal().apply(modelBuilder.getModel()), 
						features, labels, nodeIds.range(0, 0.8), null);
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : nodeIds.range(0.8, 1)) {
			Matrix nodeFeatures = features.accessRow(node).asRow();
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += output.argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/nodeIds.range(0.8, 1).size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
