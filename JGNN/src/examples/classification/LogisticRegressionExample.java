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
public class LogisticRegressionExample {

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
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
		
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new GradientDescent(0.1))
				.setEpochs(300)
				.setNumBatches(10)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(ModelTraining.Loss.L2)
				.train(modelBuilder.getModel(), features, labels, trainIds, null);
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : testIds) {
			Matrix nodeFeatures = features.accessRow(node).asRow();
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/testIds.size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
