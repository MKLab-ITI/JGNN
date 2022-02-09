package mklab.JGNN.examples.classification;

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
 * <b>Algorithms</b>   : multiplayer perceptron<br>
 * <b>Datasets</b>     : Lymphography<br>
 * <b>Demonstrates</b> : dataset parsing, {@link ModelBuilder}, {@link ModelTraining}, accuracy evaluation<br>
 * 
 * @author Emmanouil Krasanakis
 */
public class MLPExample {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		IdConverter nodes = dataset.nodes();
		Matrix labels = nodes.oneHot(dataset.labels());
		Matrix features = nodes.oneHot(dataset.features());
		
		System.out.println("Nodes\t: "+dataset.nodes().size());
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		long numHidden = 64;
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w2", new DenseMatrix(numHidden, numClasses).setToRandom().selfAdd(-0.5).selfMultiply(2./Math.sqrt(numFeatures+numHidden)))
				.param("w1", new DenseMatrix(numFeatures, numHidden).setToRandom().selfAdd(-0.5).selfMultiply(2./Math.sqrt(numHidden+numClasses)))
				.param("b1", new DenseTensor(numHidden))
				.param("b2", new DenseTensor(numClasses))
				.operation("h = relu(x@w1+b1)")
				.operation("yhat = sigmoid(h@w2+b2)")
				.out("yhat")
				.print();
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
		
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new GradientDescent(0.1))
				.setEpochs(150)
				.setNumBatches(10)
				.setParallelization(true)
				.setLoss(ModelTraining.Loss.L2)
				.train(modelBuilder.getModel(), features, labels, trainIds);
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
