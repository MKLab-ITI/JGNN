package mklab.JGNN.builders;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;

/**
 * 
 * @author Emmanouil Krasanakis
 */
public class GCNBuilder extends ModelBuilder {
	private int currentLayer;
	private long currentLayerInputDims;
	
	public GCNBuilder(Model model, Matrix adjacencyMatrix, long embeddingDims) {
		this(model, adjacencyMatrix, (Matrix) new DenseMatrix(adjacencyMatrix.getRows(), embeddingDims).setToRandom().setToNormalized());
	}
	public GCNBuilder(Model model, Matrix adjacencyMatrix, Matrix H0) {
		super(model);
		var("u");
		var("v");
		constant("W", adjacencyMatrix);
		param("H0", H0);
		currentLayer = 0;
		currentLayerInputDims = H0.getCols();
	}
	public GCNBuilder(Matrix adjacencyMatrix, long embeddingDims) {
		this(adjacencyMatrix, (Matrix) new DenseMatrix(adjacencyMatrix.getRows(), embeddingDims).setToRandom().setToNormalized());
	}
	public GCNBuilder(Matrix adjacencyMatrix, Matrix H0) {
		this(new Model(), adjacencyMatrix, H0);
	}
	protected GCNBuilder layerOperation(String formula) {
		operation(formula
					.replace("{l+1}", Integer.toString(currentLayer+1))
					.replace("{l}", Integer.toString(currentLayer)));
		return this;
	}
	public GCNBuilder layerFromFormula(String formula, long outputDims) {
		layerOperation(formula);
		currentLayer += 1;
		currentLayerInputDims = outputDims;
		return this;
	}
	
	public GCNBuilder aggregateAndTransform(String activation, long outputDims) {
		param("W"+currentLayer, new DenseMatrix(currentLayerInputDims, outputDims).setToRandom().setToNormalized());
		param("Wd"+currentLayer, new DenseMatrix(currentLayerInputDims, outputDims).setToRandom().setToNormalized());
		if(activation.equals("linear"))
			layerFromFormula("H{l+1} = W@H{l}@W{l} + H{l}@Wd{l}", outputDims);
		else if(activation.equals("lrelu"))
			layerFromFormula("H{l+1} = prelu(W@H{l}@W{l} + H{l}@Wd{l}, 0.2 )", outputDims);
		else
			layerFromFormula("H{l+1} = "+activation+"(W@H{l}@W{l} + H{l}@Wd{l})", outputDims);
		return this;
	}
	
	public GCNBuilder similarity(String method) {
		if(method.equals("distmult")) {
			param("DistMult", new DenseTensor(currentLayerInputDims).setToRandom().setToNormalized());
			layerOperation("sim = sigmoid(sum(H{l}[u]*H{l}[v]*DistMult))");
			out("sim");
		}
		else if(method.equals("dot")) {
			layerOperation("sim = sigmoid(sum(H{l}[u]*H{l}[v]))");
			out("sim");
		}
		else if(method.equals("disteuclidean")) {
			param("dimWeights", new DenseTensor(currentLayerInputDims).setToRandom().setToNormalized());
			layerOperation("diff = H{l}[u]-H{l}[v]");
			layerOperation("sim = sigmoid(-sum(diff*diff*dimWeights))");
			out("sim");
		}
		else if(method.equals("euclidean")) {
			layerOperation("diff = H{l}[u]-H{l}[v]");
			layerOperation("sim = sigmoid(-sum(diff*diff))");
			out("sim");
		}
		else if(method.equals("transe")) {
			param("transE", new DenseTensor(currentLayerInputDims).setToRandom().setToNormalized());
			layerOperation("diff = H{l}[u]-H{l}[v]-transE");
			layerOperation("sim = sigmoid(-sum(diff*diff))");
			out("sim");
		}
		else 
			throw new RuntimeException("Invalid similarity function");
		return this;
	}
}
