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

public class GNNClassification {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.Cora();
		IdConverter nodes = dataset.nodes();
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/5*2, nodeIds.size());
		List<Long> validationIds = nodeIds.subList(nodeIds.size()/5, nodeIds.size()/5*2);
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
		Matrix labels = nodes
				.oneHot(dataset.labels())
				.setDimensionName("nodes", "labels");
		//Matrix features = nodes.oneHot(dataset.features());
		Matrix features = labels
				.copy()
				.cast(Matrix.class)
				.setDimensionName("nodes", "features");
		for(Long testId : testIds)
			for(long i=0;i<labels.getCols();i++)
				features.put(testId, i, 0);
		
		Matrix adjacency = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue(), 1);
			adjacency.put(interaction.getValue(), interaction.getKey(), 1);
		}
		adjacency.setMainDiagonal(1).setToLaplacian().setDimensionName("nodes", "nodes");
		
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		long numHidden = 32;
		
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("nodes")
				.constant("A", adjacency)
				.constant("x", features)
				.param("w2", 0.0005, new DenseMatrix(numHidden, numClasses)
						.setRowName("hidden").setColName("labels")
						.setToRandom(new Uniform().setMean(0).setDeviation(1/Math.sqrt(numFeatures))))
				.param("w1", 0.0005, new DenseMatrix(numFeatures, numHidden)
						.setRowName("features").setColName("hidden")
						.setToRandom(new Uniform().setMean(0).setDeviation(1/Math.sqrt(numHidden))))
				.param("b2", new DenseTensor(numClasses).setDimensionName("labels"))
				.param("b1", new DenseTensor(numHidden).setDimensionName("hidden"))
				.operation("h = dropout(relu(A@x@w1+b1), 0.5)")
				.operation("yhat = softmax(A@h@w2+b2, col)")
				.operation("node_yhat = yhat[nodes]")
				.out("node_yhat")
				.print();

		Matrix nodeIdMatrix = new WrapCols(new DenseTensor(new Range(0, nodes.size())));
		Model model = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(3000)
				.setPatience(100)
				.setLoss(ModelTraining.Loss.CrossEntropy)
				.train(modelBuilder.getModel(), nodeIdMatrix, labels, trainIds, validationIds);
		
		nodeIdMatrix = new DenseTensor(nodeIds.iterator()).asColumn();
		double acc = 0;
		Matrix output = model.predict(nodeIdMatrix).get(0).cast(Matrix.class);
		for(Long i : Tensor.fromRange(0, testIds.size())) {
			Matrix nodeLabels = labels.accessRow((long)nodeIdMatrix.get(i)).asRow();
			acc += output.accessRow(i).argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/testIds.size());
	}
}
