package tutorial;


import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.datasets.Citeseer;
import mklab.JGNN.adhoc.parsers.LayeredBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.BinaryCrossEntropy;
import mklab.JGNN.nn.optimizers.Adam;

/**
 * This implementation covers code of the Neural Networks tutorial.
 * 
 * @author Emmanouil Krasanakis
 */
public class NN {

	public static void main(String[] args) {
		Dataset dataset = new Citeseer();
		Matrix labels = dataset.labels().setDimensionName("samples", "labels");
		Matrix features = dataset.features().setDimensionName("samples", "features");
		
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
		
		Slice sampleIds = dataset.samples().getSlice().shuffle(100);
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(3000)
				.setPatience(10)
				.setLoss(new BinaryCrossEntropy())
				.setValidationLoss(new BinaryCrossEntropy())
				.train(new XavierNormal().apply(modelBuilder.getModel()), 
						features, labels, sampleIds.range(0, 0.7), sampleIds.range(0.7, 0.8));
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : sampleIds.range(0.8, 1)) {
			Matrix nodeFeatures = features.accessRow(node).asRow();
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += output.argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/sampleIds.range(0.8, 1).size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
