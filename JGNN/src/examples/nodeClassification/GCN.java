package nodeClassification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelBuilder;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.distribution.Uniform;
import mklab.JGNN.core.loss.BinaryCrossEntropy;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Range;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.initializers.KaimingNormal;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.Adam;

public class GCN {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodes = dataset.nodes();
		
		Slice nodeIds = dataset.nodes().getIds().shuffle(100);
		Slice trainIds = nodeIds.range(0, 120);
		Slice validationIds = nodeIds.range(120, 620);
		Slice testIds = nodeIds.range(620, 1620);
		
		Matrix labels = nodes
				.oneHot(dataset.getLabels())
				.setDimensionName("nodes", "labels");
		//Matrix features = nodes.oneHot(dataset.features());
		/*Matrix features = labels
				.copy()
				.cast(Matrix.class)
				.setDimensionName("nodes", "features");*/
		Matrix features = nodes.oneHotFromBinary(dataset.getFeatures());
		
		Matrix adjacency = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue(), 1);
			adjacency.put(interaction.getValue(), interaction.getKey(), 1);
		}
		adjacency.setMainDiagonal(1).setToSymmetricNormalization().setDimensionName("nodes", "nodes");
		
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		for(Tensor row : features.accessRows()) 
			row.setToProbability();
		
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("nodes")
				.constant("A", adjacency)
				.constant("x", features)
				.config("hidden", 64)
				.config("classes", numClasses)
				.config("features", numFeatures)
				.config("reg", 0.0005)
				.operation("h0 = dropout(x, 0.5)")
				.operation("h1 = dropout(relu(A@(h0@matrix(features, hidden, reg))+vector(hidden)), 0.5)")
				.operation("h2 = dropout(relu(A@(h1@matrix(hidden, hidden, reg))+vector(hidden)), 0.5)")
				.operation("yhat = softmax(A@(h2@matrix(hidden, classes, reg))+vector(classes), row)")
				.operation("node_yhat = yhat[nodes]")
				.out("node_yhat")
				.print();
		
		System.out.println(modelBuilder.getExecutionGraphDot());
		
		new KaimingNormal().apply(modelBuilder.getModel());

		Matrix nodeIdMatrix = Tensor.fromRange(0, nodes.size()).asColumn();
		Model model = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(10)
				.setPatience(100)
				.setLoss(new BinaryCrossEntropy())
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
