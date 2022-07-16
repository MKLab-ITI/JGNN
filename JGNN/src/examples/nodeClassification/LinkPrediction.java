package nodeClassification;

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

public class LinkPrediction {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodes = dataset.nodes();
		Matrix labels = nodes.oneHot(dataset.labels());
		Matrix features = labels.toDense().cast(Matrix.class);
		
		Matrix adjacency = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue(), 1);
			adjacency.put(interaction.getValue(), interaction.getKey(), 1);
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
				.var("nodes1")
				.var("nodes2")
				.constant("A", adjacency)
				.constant("x", features)
				.param("w2", new DenseMatrix(numHidden, numClasses)
						.setDimensionName("hidden", "labels")
						.setToRandom(new Uniform().setMean(0).setDeviation(1/Math.sqrt(numFeatures))))
				.param("w1", new DenseMatrix(numFeatures, numHidden)
						.setDimensionName("features", "hidden")
						.setToRandom(new Uniform().setMean(0).setDeviation(1/Math.sqrt(numHidden))))
				.param("b2", new DenseTensor(numClasses))
				.param("b1", new DenseTensor(numHidden))
				.operation("h = relu(A@x@w1)")
				.operation("yhat = relu(A@h@w2, col)")
				.operation("edge_yhat = yhat[nodes1] * yhat[nodes1]")
				.out("edge_yhat")
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
				.setNumBatches(10)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(ModelTraining.Loss.CrossEntropy)
				.train(modelBuilder.getModel(), nodeIdMatrix, labels, trainIds, null);
		long toc = System.currentTimeMillis();
		
		double acc = 0;
		Tensor output = model.predict(new WrapCols(nodeIdMatrix)).get(0); // prediction for all nodes
		for(Long node : testIds) {
			Matrix nodeLabels = labels.accessRow(node).asRow();
			acc += output.cast(Matrix.class).accessRow(node).argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/testIds.size());
		System.out.println("Training time\t "+(toc-tic)/1000.);
	
	
	
		/*Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodeIds = new IdConverter();
		HashMap<Integer, String> nodeLabels = new HashMap<Integer, String>();
		for(Entry<String, String> interaction : dataset.getInteractions()) {
			String u = interaction.getKey();
			String v = interaction.getValue();
			nodeLabels.put(nodeIds.getOrCreateId(u), dataset.getLabel(u));
			nodeLabels.put(nodeIds.getOrCreateId(v), dataset.getLabel(v));
		}
		RelationalGCN gcn = new RelationalGCN(RelationalGCN.trueres_linear,
											  nodeIds.size(),//oneHot(nodeLabels), 
											  new RepeatTensor(16, 2));
		for(Entry<String, String> interaction : dataset.getInteractions()) 
			gcn.addEdge(nodeIds.getId(interaction.getKey()), nodeIds.getId(interaction.getValue()));
		gcn.trainRelational(new Regularization(new Adam(0.001), 5.E-4), 200, 0.2);*/
	}
}
