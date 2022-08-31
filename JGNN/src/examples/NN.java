

import mklab.JGNN.adhoc.builders.LayeredBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelBuilder;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.loss.Accuracy;
import mklab.JGNN.core.loss.BinaryCrossEntropy;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.Adam;


public class NN {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		Matrix labels = dataset.nodes().setDimensionName("nodes", "labels").oneHot(dataset.getLabels());
		Matrix features = dataset.nodes().setDimensionName("nodes", "features").oneHot(dataset.getFeatures());
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		ModelBuilder modelBuilder = new LayeredBuilder()
				.config("features", numFeatures)
				.config("classes", numClasses)
				.config("hidden", 64)
				.config("2hidden", 2*64)
				.layer("h{l+1} = relu(h{l}@matrix(features, hidden)+vector(hidden))")
				.layerRepeat("h{l+1} = relu(h{l}@matrix(hidden, hidden)+vector(hidden))", 2)
				.concat(2)
				.layer("yhat = softmax(h{l}@matrix(2hidden, classes)+vector(classes), row)")
				.out("yhat");
		
		Slice nodeIds = dataset.nodes().getIds().shuffle(100);
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(3000)
				.setPatience(10)
				.setLoss(new BinaryCrossEntropy())
				.setValidationLoss(new Accuracy())
				.train(new XavierNormal().apply(modelBuilder.getModel()), 
						features, labels, nodeIds.range(0, 0.7), nodeIds.range(0.7, 0.8));
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
