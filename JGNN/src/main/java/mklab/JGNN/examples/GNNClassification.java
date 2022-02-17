package mklab.JGNN.examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.ModelTraining;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.tensor.AccessSubtensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.tensor.RepeatTensor;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.models.relational.ClassificationGCN;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;
import mklab.JGNN.nn.optimizers.GradientDescent;
import mklab.JGNN.nn.optimizers.Regularization;

public class GNNClassification {

	private static final ModelBuilder MODEL_BUILDER = new ModelBuilder().var("x");

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodes = dataset.nodes();
		Matrix labels = nodes.oneHot(dataset.labels()).asTransposed();
		Matrix features = (Matrix) new DenseMatrix(10, nodes.size()).setToRandom();//nodes.oneHot(dataset.features()).asTransposed();
		Matrix adj = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> edge : dataset.getInteractions()) {
			adj.put(edge.getKey(), edge.getValue(), 1.);
			adj.put(edge.getValue(), edge.getKey(), 1.);
		}
		adj = adj.laplacian();
		
		System.out.println("Nodes\t: "+dataset.nodes().size());
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		long numFeatures = features.getRows();
		long numClasses = labels.getRows();
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.var("A")
				.var("nodes")
				.param("w1", new DenseMatrix(numClasses, numFeatures).setToRandom().selfAdd(-0.5).selfMultiply(1./Math.sqrt(numFeatures+numClasses)))
				.param("w2", new DenseMatrix(numClasses, numFeatures).setToRandom().selfAdd(-0.5).selfMultiply(1./Math.sqrt(numFeatures+numClasses)))
				.operation("h = transpose(w1@x+w2@x@A)")
				.operation("yhat = h[nodes]")
				.out("yhat")
				.print();
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
	
		BatchOptimizer optimizer = new BatchOptimizer(new Regularization(new Adam(0.01), 5.E-4));
		for(int epoch=0;epoch<150;epoch++) {
			modelBuilder.getModel().trainL2(optimizer, 
					Arrays.asList(features, adj, new DenseTensor(trainIds.iterator())), 
					Arrays.asList(new WrapCols(features.accessColumns(trainIds))));
			optimizer.updateAll();
			
			Tensor output = modelBuilder.getModel().predict(Arrays.asList(features, adj, new DenseTensor(testIds.iterator()))).get(0);
			double acc = 0;
			int i = 0;
			for(Long node : testIds) {
				acc += (((Matrix)output).accessRow(i).argmax()==labels.accessCol(node).argmax()?1:0);
				i++;
			}
			System.out.println("Acc\t "+acc/testIds.size());
		}
	}
}
