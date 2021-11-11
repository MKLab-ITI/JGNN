package mklab.JGNN.examples;

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
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.optimizers.Adam;

public class MLP {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		IdConverter nodes = dataset.nodes();
		Matrix labels = nodes.oneHot(dataset.labels()).asTransposed();
		Matrix features = nodes.oneHot(dataset.features()).asTransposed();
		
		System.out.println("Nodes\t: "+dataset.nodes().size());
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		long numFeatures = features.getRows();
		long numClasses = labels.getRows();
		long numHidden = 64;
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w2", new DenseMatrix(numClasses, numHidden).setToRandom().selfAdd(-0.5).selfMultiply(2./Math.sqrt(numFeatures+numHidden)))
				.param("w1", new DenseMatrix(numHidden, numFeatures).setToRandom().selfAdd(-0.5).selfMultiply(2./Math.sqrt(numHidden+numClasses)))
				.operation("h = relu(w1@x)")
				.operation("yhat = sigmoid(w2@h)")
				.out("yhat")
				.print();
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(150)
				.setNumBatches(10)
				.setParallelization(true)
				.setLoss(ModelTraining.Loss.L2)
				.train(modelBuilder.getModel(), features, labels, trainIds);
		long toc = System.currentTimeMillis();

		double acc = 0;
		for(Long node : testIds) {
			Matrix nodeFeatures = features.accessCol(node).asColumn();
			Matrix nodeLabels = labels.accessCol(node).asColumn();
			Tensor output = model.predict(nodeFeatures).get(0);
			acc += (output.argmax()==nodeLabels.argmax()?1:0);
		}
		System.out.println("Acc\t "+acc/testIds.size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}

}
