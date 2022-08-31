import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelBuilder;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.nn.Optimizer;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.loss.BinaryCrossEntropy;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.Adam;

public class Introduction {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		IdConverter nodeIdConverter = dataset.nodes();
		Matrix labels = nodeIdConverter.oneHot(dataset.getLabels()).setDimensionName("samples", "classes");
		Matrix features = nodeIdConverter.oneHot(dataset.getFeatures()).setDimensionName("samples", "features");

		long numFeatures = features.getCols();
		long numClasses = labels.getCols();

		Slice nodeIds = dataset.nodes().getIds().shuffle();
		ModelTraining trainer = new ModelTraining()
			.setOptimizer(new Adam(0.1))
			.setEpochs(3000)
			.setPatience(100)
			.setNumBatches(10)
			.setParallelizedStochasticGradientDescent(true)
			.setLoss(new BinaryCrossEntropy());
		
		ModelBuilder modelBuilder = new ModelBuilder()
				.config("features", numFeatures)
				.config("classes", numClasses)
				.config("regularize", 1.E-5)
				.var("x")
				.operation("h = relu(x@matrix(features, 64, regularize)+vector(64))")
				.operation("yhat = softmax(h@matrix(64, classes)+vector(classes), row)")
				.out("yhat")
				.assertBackwardValidity();
		
		System.out.println(modelBuilder.getExecutionGraphDot());
		Model model = modelBuilder
				.getModel()
				.init(new XavierNormal())
				.train(trainer, features, labels, 
						nodeIds.range(0, 0.5), 
						nodeIds.range(0.5, 0.75));
		
		double acc = 0;
		for(Long node : nodeIds.range(0.75, 1)) {
			Matrix nodeFeatures = features.accessRow(node).asRow();
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/nodeIds.range(0.75, 1).size());
		
	}

}
