package mklab.JGNN.examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.optimizers.BatchOptimizer;
import mklab.JGNN.nn.optimizers.GradientDescent;
import mklab.JGNN.nn.optimizers.Regularization;

public class Classification {

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
		int dims = 2;
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w1", new DenseMatrix(numClasses, numFeatures).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(1./dims)))
				.operation("yhat = sigmoid(w1@x)")
				.out("yhat")
				.print();
		
		BatchOptimizer optimizer = new BatchOptimizer(new Regularization(new GradientDescent(0.1), 0.001));
		Model model = modelBuilder.getModel();
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds);
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		Matrix trainFeatures = new WrapCols(features.accessColumns(trainIds));
		Matrix trainLabels = new WrapCols(labels.accessColumns(trainIds));
		System.out.println("Train features: "+trainFeatures.describe());
		
		for(int epoch=0;epoch<150;epoch++) {
			System.out.print("Epoch "+epoch);
			Tensor errors = 
					model.trainSampleDifference(optimizer, Arrays.asList(trainFeatures), Arrays.asList(trainLabels))
					.get(0)
					.subtract(trainLabels);
			
			double acc = 0;
			for(Long node : testIds) {
				Matrix nodeFeatures = features.accessCol(node).asColumn();
				Matrix nodeLabels = labels.accessCol(node).asColumn();
				Tensor output = modelBuilder.getModel().predict(nodeFeatures).get(0);
				acc += (output.argmax()==nodeLabels.argmax()?1:0);
			}
			optimizer.updateAll();
			System.out.print("\t error "+errors.abs().sum()/trainLabels.size());
			System.out.println("\t accuracy "+acc/testIds.size());
		}
	}

}
