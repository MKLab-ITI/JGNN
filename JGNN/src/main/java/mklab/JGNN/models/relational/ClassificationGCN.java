package mklab.JGNN.models.relational;

import java.util.Arrays;
import java.util.List;

import mklab.JGNN.builders.GCNBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.measures.AUC;
import mklab.JGNN.measures.Accuracy;
import mklab.JGNN.measures.Precision;

/**
 * Uses a {@link GCNBuilder} to construct a Graph Convolutional Network
 * for unsupervised link prediction.
 * 
 * @author Emmanouil Krasanakis
 */
public class ClassificationGCN extends Model {
	private GCNBuilder builder;
	public ClassificationGCN(Matrix H0, Tensor layerDims, int layerTransformationDepth) {
		builder = new GCNBuilder(this, new SparseMatrix(H0.getRows(), H0.getRows()), H0, (int)layerDims.get(0));
		/*for(int i=1;i<layerDims.size();i++) {
			builder.aggregateAndTransform("tanh", (int)layerDims.get(i));
			for(int transform=0;transform<layerTransformationDepth;transform++)
				builder.transform("tanh", (int)layerDims.get(i));
		}
		builder
			.multiclass(H0.getCols())
			.assertBackwardValidity();*/
	}
	
	public void addEdge(int i, int j) {
		if(builder.getAdjacencyMatrix().get(i, j)!=0)
			return;
		builder.getAdjacencyMatrix().put(i, j, 1);
		builder.getAdjacencyMatrix().put(j, i, 1);
	}
	
	public double predict(int u, int v) {
		return predict(Arrays.asList(DenseTensor.fromDouble(u), DenseTensor.fromDouble(v)))
							.get(0) //first output
							.get(0);//first of the three tensor elements
	}
	
	public void trainClassification(Optimizer optimizer, int epochs,
			Tensor trainingNodes, List<Tensor> trainingLabels, Tensor testNodes, List<Tensor> testLabels) {
		builder.getAdjacencyMatrix().setToLaplacian();
		if(testNodes!=null) {
			System.out.println("Number of nodes: "+builder.getAdjacencyMatrix().getRows());
			System.out.println("Number of edges: "+builder.getAdjacencyMatrix().getNumNonZeroElements());
		}
		for(int epoch=0;epoch<epochs;epoch++) {
			if(testNodes!=null)
				System.out.print("Epoch "+epoch);
			trainSample(optimizer, Arrays.asList(trainingNodes), 
					Arrays.asList(new WrapCols(trainingLabels)));
			if(testNodes!=null)  {
				List<Tensor> outputs = ((Matrix)predict(Arrays.asList(testNodes)).get(0)).accessColumns();
				System.out.println(
								   " | test AUC "+new AUC().evaluate(outputs, testLabels)
								  +" | test prec "+new Precision().evaluate(outputs, testLabels)
								  +" | test acc "+new Accuracy().evaluate(outputs, testLabels));
			}
		}
	}
	
}
