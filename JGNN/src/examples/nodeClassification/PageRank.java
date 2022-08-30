package nodeClassification;

import java.util.Map.Entry;

import mklab.JGNN.adhoc.builders.GraphFilterBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;


public class PageRank {
	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		Slice nodes = dataset.nodes().getIds().shuffle(100);
		Slice test = nodes.range(0.2, 1);
		Matrix labels = dataset
				.nodes()
				.setDimensionName("nodes", "labels")
				.oneHot(dataset.getLabels());
		Matrix features = labels.copy().cast(Matrix.class);
		for(Long testId : test)
			for(long i=0;i<labels.getCols();i++) 
				features.put(testId, i, 0);
		
		Matrix adjacency = new SparseMatrix(nodes.size(), nodes.size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue(), 1);
			adjacency.put(interaction.getValue(), interaction.getKey(), 1);
		}
		adjacency.setToSymmetricNormalization().setDimensionName("nodes", "nodes");
		
		System.out.println("Labels   : "+labels.describe());
		System.out.println("Features : "+features.describe());
		System.out.println("Adjacency: "+adjacency.describe());
		
		GraphFilterBuilder pagerank = new GraphFilterBuilder(adjacency, features);
		for(int i=0;i<10;i++)
			pagerank.addShift(Math.pow(0.9, i));

		System.out.println(pagerank.getExecutionGraphDot());

		Matrix nodeIdMatrix = nodes.asTensor().asColumn();
		double acc = 0;
		Matrix output = pagerank.getModel().predict(nodeIdMatrix).get(0).cast(Matrix.class);
		for(Long i : test) {
			Matrix nodeLabels = labels.accessRow((long)nodeIdMatrix.get(i)).asRow();
			acc += output.accessRow(i).argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Accuracy "+acc/test.size());
	}
}
