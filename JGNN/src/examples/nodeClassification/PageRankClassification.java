package nodeClassification;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;

public class PageRankClassification {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodes = dataset.nodes();
		
		ArrayList<Long> nodeIds = dataset.nodes().getIds();
		Collections.shuffle(nodeIds, new Random(100));
		List<Long> testIds = nodeIds.subList(0, nodeIds.size()/5);
		
		Matrix labels = nodes
				.oneHot(dataset.labels())
				.setDimensionName("nodes", "labels");
		Matrix features = labels
				.copy()
				.cast(Matrix.class);
		for(Long testId : testIds)
			for(long i=0;i<labels.getCols();i++)
				features.put(testId, i, 0);
		
		Matrix adjacency = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue(), 1);
			adjacency.put(interaction.getValue(), interaction.getKey(), 1);
		}
		for(Long node : nodes.getIds())
			adjacency.put(node, node, 1);
		adjacency.setToLaplacian().setDimensionName("nodes", "nodes");
		
		System.out.println("Labels   : "+labels.describe());
		System.out.println("Features : "+features.describe());
		System.out.println("Adjacency: "+adjacency.describe());
		
		ModelBuilder pagerank = new ModelBuilder()
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
				.out("node_yhat")
				.print();
		
		
		Matrix nodeIdMatrix = new DenseTensor(nodeIds.iterator()).asColumn();
		double acc = 0;
		Matrix output = pagerank.getModel().predict(nodeIdMatrix).get(0).cast(Matrix.class);
		for(Long i : Tensor.fromRange(0, testIds.size())) {
			Matrix nodeLabels = labels.accessRow((long)nodeIdMatrix.get(i)).asRow();
			acc += output.accessRow(i).argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/testIds.size());
	}
}
