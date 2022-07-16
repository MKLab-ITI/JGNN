package classification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.ModelTraining;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.GradientDescent;

/**
 * <b>Tasks</b>	       : node classification<br>
 * <b>Algorithms</b>   : multilayer perceptron<br>
 * <b>Datasets</b>     : Lymphography<br>
 * <b>Demonstrates</b> : dataset parsing, {@link ModelBuilder}, {@link ModelTraining}, accuracy evaluation, named dimensions, automatic parsing, {@link ModelInitializer}<br>
 * 
 * @author Emmanouil Krasanakis
 */
public class MLPExample {

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
				.operation("h = relu(x@matrix(features, 64)+vector(64))")
				.operation("yhat = sigmoid(h@matrix(64, classes)+vector(classes))")
				.out("yhat")
				.print();
		System.out.println(modelBuilder.getExecutionGraphDot());
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
		
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new GradientDescent(0.01))
				.setEpochs(3000)
				.setPatience(100)
				.setNumBatches(10)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(ModelTraining.Loss.CrossEntropy)
				.train(new XavierNormal().apply(modelBuilder.getModel()), 
						features, labels, trainIds, null);
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : testIds) {
			Matrix nodeFeatures = features.accessDim(node, "nodes").asRow();
			Matrix nodeLabels = labels.accessDim(node, "nodes").asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/testIds.size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
