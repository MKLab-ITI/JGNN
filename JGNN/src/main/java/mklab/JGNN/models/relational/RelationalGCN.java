package mklab.JGNN.models.relational;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.builders.GCNBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
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
public class RelationalGCN extends Model {
	public static String trueres_tanh = "H{l+1} = tanh(W@H{l}@W{l} + H{l}@R{l} + b{l})";
	public static String trueres_linear = "H{l+1} = W@H{l}@W{l} + H{l}@R{l} + b{l}";
	public static String trueres_relu = "H{l+1} = relu(W@H{l}@W{l} + H{l}@R{l} + b{l})";
	public static String renormalization_relu = "H{l+1} = relu((W@H{l}+H{l})@W{l} + b{l})";
	public static String renormalization_softmax_relu = "H{l+1} = relu((W@(max(H{l})*H{l})+H{l})@W{l} + b{l})";
	public static String renormalization_linear = "H{l+1} = (W@H{l}+H{l})@W{l} + b{l}";
	public static String renormalization_tanh = "H{l+1} = tanh((W@H{l}+H{l})@W{l} + b{l})";
	
	private GCNBuilder builder;
	public RelationalGCN(String formula, long numNodes, Tensor layerDims) {
		builder = new GCNBuilder(this, new SparseMatrix(numNodes, numNodes), (long)layerDims.get(0));
		for(int i=1;i<layerDims.size();i++) 
			builder.addGCNLayer(formula, (long)layerDims.get(i));
		builder
			.similarity("distmult")
			.assertForwardValidity(Arrays.asList(3, 3))
			.assertBackwardValidity();
	}
	public RelationalGCN(String formula, Matrix H0, Tensor layerDims, int layerTransformationDepth) {
		builder = new GCNBuilder(this, new SparseMatrix(H0.getRows(), H0.getRows()), H0, (int)layerDims.get(0));
		for(int i=1;i<layerDims.size();i++) 
			builder.addGCNLayer(formula, (int)layerDims.get(i));
		builder
			.similarity("distmult")
			.assertForwardValidity(Arrays.asList(3, 3))
			.assertBackwardValidity();
	}
	
	public void addEdge(int i, int j) {
		if(builder.getAdjacencyMatrix().get(i, j)!=0)
			return;
		builder.getAdjacencyMatrix().put(i, j, 1);
		builder.getAdjacencyMatrix().put(j, i, 1);
		builder.getAdjacencyMatrix().put(i, i, 1);
		builder.getAdjacencyMatrix().put(j, j, 1);
	}
	
	public double predict(int u, int v) {
		return predict(Arrays.asList(DenseTensor.fromDouble(u), DenseTensor.fromDouble(v)))
							.get(0) //first output
							.get(0);//first of the three tensor elements
	}
	
	public void trainRelational(Optimizer optimizer, int epochs, double fractionOfTest) {
		Matrix W = builder.getAdjacencyMatrix();
		Matrix testMatrix = fractionOfTest==0?null:builder.getAdjacencyMatrix().zeroCopy();
		for(Entry<Long, Long> entry : W.getNonZeroEntries()) {
			if(Math.random()<fractionOfTest) {
				long row = entry.getKey();
				long col = entry.getValue();
				testMatrix.put(row, col, W.get(row, col));
				testMatrix.put(col, row, W.get(col, row));
				W.put(row, col, 0);
				W.put(row, col, 0);
			}
		}
		W.setToLaplacian();
		if(testMatrix!=null) {
			//print();
			System.out.println("Number of nodes: "+W.getRows());
			System.out.println("Number of edges: "+W.getNumNonZeroElements());
		}
		for(int epoch=0;epoch<epochs;epoch++) {
			if(testMatrix!=null)
				System.out.print("Epoch "+epoch);
			RelationalData trainingData = new RelationalData((SparseMatrix) W, RelationalData.NegativeSamplingType.PERMUTATION);
			trainSample(optimizer, trainingData.getInputs(), trainingData.getOutputs());
			if(testMatrix!=null)  {
				RelationalData testData = new RelationalData((SparseMatrix) testMatrix, W, 
						RelationalData.NegativeSamplingType.RANDOM);
				List<Tensor> outputs = predict(testData.getInputs());
				System.out.println(
								   " | test AUC "+new AUC().evaluate(outputs, testData.getOutputs())
								  +" | test prec "+new Precision().evaluate(outputs, testData.getOutputs())
								  +" | test acc "+new Accuracy().evaluate(outputs, testData.getOutputs()));
			}
		}
	}
	
}
