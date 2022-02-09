package mklab.JGNN.examples.nodeClassification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.ModelTraining;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.distribution.Uniform;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Range;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.Regularization;

public class GNNClassification {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodes = dataset.nodes();
		Matrix labels = nodes.oneHot(dataset.labels());
		Matrix features = labels.toDense().cast(Matrix.class);//nodes.oneHot(dataset.features());
		
		
		Matrix adjacency = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue());
			adjacency.put(interaction.getValue(), interaction.getKey());
		}
		for(Long node : nodes.getIds())
			adjacency.put(node, node, 1);
		adjacency.setToLaplacian().setRowName("nodes").setColName("nodes");
		features.setRowName("nodes").setColName("features");
		labels.setRowName("nodes").setColName("labels");
		
		System.out.println("Nodes\t: "+dataset.nodes().size());
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		long numHidden = 32;
		
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("nodes")
				.constant("A", adjacency)
				.constant("x", features)
				.param("w2", new DenseMatrix(numHidden, numClasses)
						.setRowName("hidden").setColName("labels")
						.setToRandom(new Uniform().setMean(0).setDeviation(Math.sqrt(24)/Math.sqrt(numFeatures))))
				.param("w1", new DenseMatrix(numFeatures, numHidden)
						.setRowName("features").setColName("hidden")
						.setToRandom(new Uniform().setMean(0).setDeviation(Math.sqrt(24)/Math.sqrt(numHidden))))
				.param("b2", new DenseTensor(numClasses))
				.param("b1", new DenseTensor(numHidden))
				.operation("h = relu(A@x@w1+b1)")
				.operation("yhat = softmax(A@h@w2+b2, col)")
				//.operation("yhat = A@x@w1")
				.operation("node_yhat = yhat[nodes]")
				.out("node_yhat")
				.print();
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size());
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
		Matrix nodeIdMatrix = new WrapCols(new DenseTensor(new Range(0, nodes.size())));
		long tic = System.currentTimeMillis();
		Model model = new ModelTraining()
				.setOptimizer(new Regularization(new Adam(0.01), 0.0005))
				.setEpochs(300)
				.setNumBatches(1)
				.setParallelization(false)
				.setLoss(ModelTraining.Loss.CrossEntropy)
				.train(modelBuilder.getModel(), nodeIdMatrix, labels, trainIds);
		long toc = System.currentTimeMillis();
		
		double acc = 0;
		Tensor output = model.predict(new WrapCols(nodeIdMatrix)).get(0); // prediction for all nodes
		for(Long node : testIds) {
			Matrix nodeLabels = labels.accessRow(node).asRow();
			acc += output.cast(Matrix.class).accessRow(node).argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/testIds.size());
		System.out.println("Time\t "+(toc-tic)/1000.);
	}
}
