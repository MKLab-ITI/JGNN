package classification;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelBuilder;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.loss.Accuracy;
import mklab.JGNN.core.loss.BinaryCrossEntropy;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.optimizers.GradientDescent;

/**
 * <b>Tasks</b>	       : node classification<br>
 * <b>Algorithms</b>   : logistic regression<br>
 * <b>Datasets</b>     : Lymphography<br>
 * <b>Demonstrates</b> : dataset parsing, {@link ModelBuilder}, {@link ModelTraining}, accuracy evaluation<br>
 * 
 * @author Emmanouil Krasanakis
 */
public class LogisticRegression {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		IdConverter nodes = dataset.nodes();
		Matrix labels = nodes.oneHot(dataset.getLabels()).setDimensionName("samples", "labels");
		Matrix features = nodes.oneHot(dataset.getFeatures()).setDimensionName("samples", "features");
		
		System.out.println("Nodes\t: "+dataset.nodes().size());
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w", new DenseMatrix(numFeatures, numClasses)
						.setDimensionName("features", "labels")
						.setToRandom().selfAdd(-0.5).selfMultiply(1./Math.sqrt(numFeatures+numClasses)))
				.param("b", new DenseTensor(numClasses).setDimensionName("labels"))
				.operation("yhat = sigmoid(x@w)")
				.out("yhat")
				.print();
		
		Slice nodeIds = dataset.nodes().getIds().shuffle(100);
		
		
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new GradientDescent(0.01))
				.setEpochs(600)
				.setNumBatches(10)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(new BinaryCrossEntropy())
				.setValidationLoss(new Accuracy())
				.train(modelBuilder.getModel(), features, labels, nodeIds.range(0, 0.6), nodeIds.range(0.6, 0.8));
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : nodeIds.range(0.8, 1)) {
			Matrix nodeFeatures = features.accessRow(node).asRow();
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/nodeIds.range(0.8, 1).size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
