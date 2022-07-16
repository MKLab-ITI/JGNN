import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.ModelTraining;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.Adam;

public class Introduction {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		IdConverter nodeIdConverter = dataset.nodes();
		Matrix labels = nodeIdConverter.oneHot(dataset.getLabels()).setDimensionName("samples", "classes");
		Matrix features = nodeIdConverter.oneHot(dataset.getFeatures()).setDimensionName("samples", "features");

		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		
		ModelBuilder modelBuilder = new ModelBuilder()
				.config("features", numFeatures)
				.config("classes", numClasses)
				.config("regularize", 1.E-5)
				.var("x")
				.operation("h = relu(x@matrix(features, 64, regularize)+vector(64))")
				.operation("yhat = softmax(h@matrix(64, classes)+vector(classes), row)")
				.out("yhat");
		
		System.out.println(modelBuilder.getExecutionGraphDot());
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds);
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/2, nodeIds.size());
		List<Long> validationIds = nodeIds.subList(nodeIds.size()/4, nodeIds.size()/2);
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/4);
		
		Model model = new XavierNormal().apply(modelBuilder.getModel());
		
		Optimizer optimizer = new Adam(0.1);

		ModelTraining trainer = new ModelTraining()
			.setOptimizer(optimizer)
			.setEpochs(3000)
			.setPatience(100)
			.setNumBatches(10)
			.setParallelizedStochasticGradientDescent(true)
			.setLoss(ModelTraining.Loss.CrossEntropy);
		
		model = trainer.train(model, features, labels, trainIds, validationIds);
		
		double acc = 0;
		for(Long node : testIds) {
			Matrix nodeFeatures = features.accessRow(node).asRow();
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/testIds.size());
		
	}

}
