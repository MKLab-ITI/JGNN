package nodeClassification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import mklab.JGNN.builders.GraphFilterBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.ModelTraining;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.GradientDescent;

public class PageRankClassification {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.Cora();
		IdConverter nodes = dataset.nodes();
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/2);
		List<Long> trainIds = nodeIds.subList(nodeIds.size()/2, nodeIds.size());
		//List<Long> featureIds = nodeIds.subList(trainIds.size()/2, nodeIds.size());
		trainIds = nodeIds.subList(0, trainIds.size()/2);
		
		
		Matrix labels = nodes.setDimensionName("nodes", "labels").oneHot(dataset.getLabels());
		Matrix features = labels.copy().cast(Matrix.class);
		for(Long testId : testIds)
			for(long i=0;i<labels.getCols();i++)
				features.put(testId, i, 0);
		for(Long testId : trainIds)
			for(long i=0;i<labels.getCols();i++)
				features.put(testId, i, 0);
		
		
		Matrix adjacency = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue(), 1);
			adjacency.put(interaction.getValue(), interaction.getKey(), 1);
		}
		adjacency.setToLaplacian().setDimensionName("nodes", "nodes");
		
		System.out.println("Labels   : "+labels.describe());
		System.out.println("Features : "+features.describe());
		System.out.println("Adjacency: "+adjacency.describe());
		
		/*ModelBuilder pagerank = new ModelBuilder()
				.var("nodes")
				.constant("A", adjacency)
				.constant("x", features)
				.constant("a", 0.9)
				.operation("h1 = a*A@x + (1-a)*x")
				.operation("h2 = a*A@h1 + (1-a)*x")
				.operation("h3 = a*A@h2 + (1-a)*x")
				.operation("h4 = a*A@h3 + (1-a)*x")
				.operation("h5 = a*A@h4 + (1-a)*x")
				.operation("h6 = a*A@h5 + (1-a)*x")
				.operation("h7 = a*A@h6 + (1-a)*x")
				.operation("h8 = a*A@h7 + (1-a)*x")
				.operation("h9 = a*A@h8 + (1-a)*x")
				.operation("yhat = a*A@h9 + (1-a)*x")
				.operation("node_yhat = yhat[nodes]")
				.out("node_yhat");*/
		ModelBuilder pagerank = new GraphFilterBuilder(adjacency, features)
				//.addEmbeddingLayer(16)
				.addLayer(1)
				.addLayer(0.85)
				.addLayer(Math.pow(0.85,2))
				.addLayer(Math.pow(0.85,3))
				.addLayer(Math.pow(0.85,4))
				.addLayer(Math.pow(0.85,5))
				.addLayer(Math.pow(0.85,6))
				.addLayer(Math.pow(0.85,7))
				.addLayer(Math.pow(0.85,8))
				.addLayer(Math.pow(0.85,9))
				;

		System.out.println(pagerank.getExecutionGraphDot());

		Matrix nodeIdMatrix = new DenseTensor(nodeIds.iterator()).asColumn();
		/*new ModelTraining()
				.setOptimizer(new Adam(1))
				.setEpochs(30)
				.setPatience(100)
				.setParallelizedStochasticGradientDescent(true)
				.setLoss(ModelTraining.Loss.L2)
				.train(new XavierNormal().apply(pagerank.getModel()), 
						nodeIdMatrix, labels, trainIds, null);*/
		double acc = 0;
		Matrix output = pagerank.getModel().predict(nodeIdMatrix).get(0).cast(Matrix.class);
		for(Long i : Tensor.fromRange(0, testIds.size())) {
			Matrix nodeLabels = labels.accessRow((long)nodeIdMatrix.get(i)).asRow();
			acc += output.accessRow(i).argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/testIds.size());
	}
}
