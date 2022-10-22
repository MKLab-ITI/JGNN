

import java.util.ArrayList;
import java.util.Arrays;

import mklab.JGNN.adhoc.IdConverter;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.parsers.LayeredBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.nn.Loss;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.CategoricalCrossEntropy;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;

public class GraphClassification {

	public static void main(String[] args) throws Exception {
		IdConverter nodeLabelIds = new IdConverter();
		nodeLabelIds.getOrCreateId("A");
		nodeLabelIds.getOrCreateId("B");
		nodeLabelIds.getOrCreateId("C");
		
		IdConverter graphLabelIds = new IdConverter();
		graphLabelIds.getOrCreateId("0");
		graphLabelIds.getOrCreateId("1");
		
		
		ArrayList<Matrix> graphMatrices = new ArrayList<Matrix>();
		ArrayList<Matrix> nodeFeatures = new ArrayList<Matrix>();
		ArrayList<Tensor> graphLabels = new ArrayList<Tensor>();
		
		/**
		 * CREATING DATA
		 */
		graphMatrices.add(new SparseMatrix(3, 3)
				.put(0, 1, 1).put(1, 0, 1)
				.put(1, 2, 1).put(2, 1, 1)
				.put(2, 0, 1).put(0, 2, 1));
		nodeFeatures.add(new SparseMatrix(3, nodeLabelIds.size())
				.put(0, nodeLabelIds.getId("A"), 1)
				.put(1, nodeLabelIds.getId("B"), 1)
				.put(2, nodeLabelIds.getId("B"), 1));
		graphLabels.add(new DenseTensor(graphLabelIds.size()).put(graphLabelIds.getId("0"), 1));
		

		graphMatrices.add(new SparseMatrix(3, 3)
				.put(0, 1, 1).put(1, 0, 1)
				.put(2, 0, 1).put(0, 2, 1));
		nodeFeatures.add(new SparseMatrix(3, nodeLabelIds.size())
				.put(0, nodeLabelIds.getId("B"), 1)
				.put(1, nodeLabelIds.getId("A"), 1)
				.put(2, nodeLabelIds.getId("B"), 1));
		graphLabels.add(new DenseTensor(graphLabelIds.size()).put(graphLabelIds.getId("0"), 1));
		
		
		graphMatrices.add(new SparseMatrix(3, 3)
				.put(0, 1, 1).put(1, 0, 1)
				.put(1, 2, 1).put(2, 1, 1)
				.put(2, 0, 1).put(0, 2, 1));
		nodeFeatures.add(new SparseMatrix(3, nodeLabelIds.size())
				.put(0, nodeLabelIds.getId("A"), 1)
				.put(1, nodeLabelIds.getId("C"), 1)
				.put(2, nodeLabelIds.getId("C"), 1));
		graphLabels.add(new DenseTensor(graphLabelIds.size()).put(graphLabelIds.getId("1"), 1));
		

		graphMatrices.add(new SparseMatrix(3, 3)
				.put(0, 1, 1).put(1, 0, 1)
				.put(1, 2, 1).put(2, 1, 1));
		nodeFeatures.add(new SparseMatrix(3, nodeLabelIds.size())
				.put(0, nodeLabelIds.getId("A"), 1)
				.put(1, nodeLabelIds.getId("A"), 1)
				.put(2, nodeLabelIds.getId("C"), 1));
		graphLabels.add(new DenseTensor(graphLabelIds.size()).put(graphLabelIds.getId("1"), 1));
		

		/**
		 * DEFINING THE ARCHITECTURE 
		 */	
		ModelBuilder builder = new LayeredBuilder()
			    .var("A")  
			    .config("features", nodeLabelIds.size())
			    .config("classes", graphLabelIds.size())
			    .config("reduced", 2)
			    .config("hidden", 4)
			    .layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden)))") 
			    .layer("h{l+1}=relu(A@(h{l}@matrix(hidden, hidden)))")
			    .operation("s{l}=sort(h{l}, reduced)")
			    //.layer("h{l+1}=reshape(h{l}[s{l}],1,reducedFeatures)@matrix(reducedFeatures, classes)")
			    .layer("h{l+1}=softmax(sum(h{l}[s{l}]@matrix(hidden, classes), row))")
			    .out("h{l}");
		System.out.println(builder.getExecutionGraphDot());
		
		Model model = builder.getModel().init(new XavierNormal());
		BatchOptimizer optimizer = new BatchOptimizer(new Adam(0.01));
		Loss loss = new CategoricalCrossEntropy();
		for(int epoch=0; epoch<300; epoch++) {
			// gradient update
			for(int graphId=0; graphId<graphLabels.size(); graphId++) {
			     Matrix adjacency = graphMatrices.get(graphId).setDimensionName("nodes", "nodes");
			     Matrix features = nodeFeatures.get(graphId).setDimensionName("nodes", "features");
			     Tensor graphLabel = graphLabels.get(graphId).setDimensionName("classes").asRow(); 
			     model.train(loss, optimizer, 
			          Arrays.asList(features, adjacency), 
			          Arrays.asList(graphLabel));
			}
			optimizer.updateAll();
			// measure accuracy (on train data)
			double acc = 0;
			for(int graphId=0; graphId<graphLabels.size(); graphId++) {
			     Matrix adjacency = graphMatrices.get(graphId);
			     Matrix features= nodeFeatures.get(graphId);
			     Tensor graphLabel = graphLabels.get(graphId); 
			     if(model.predict(Arrays.asList(features, adjacency)).get(0).argmax()==graphLabel.argmax())
			    	 acc += 1;
			}
			System.out.println(acc/graphLabels.size());
		}
		
	}
}
