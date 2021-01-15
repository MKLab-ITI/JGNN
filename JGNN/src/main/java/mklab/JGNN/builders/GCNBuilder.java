package mklab.JGNN.builders;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.distribution.Uniform;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;

/**
 * 
 * @author Emmanouil Krasanakis
 */
public class GCNBuilder extends ModelBuilder {
	private int currentLayer;
	private long currentLayerInputDims;
	private Matrix adjacencyMatrix;
	
	public Matrix getAdjacencyMatrix() {
		return adjacencyMatrix;
	}
	
	public GCNBuilder switchAdjacencyMatrix(Matrix adjacencyMatrix) {
		this.adjacencyMatrix = adjacencyMatrix;
		constant("W", adjacencyMatrix);
		return this;
	}
	
	public GCNBuilder(Model model, Matrix adjacencyMatrix, long embeddingDims) {
		this(model, adjacencyMatrix, null, adjacencyMatrix.getRows());
	}
	public GCNBuilder(Model model, Matrix adjacencyMatrix, Matrix nodeFeatures) {
		this(model, adjacencyMatrix, nodeFeatures, 0);
	}
	
	public GCNBuilder(Model model, Matrix adjacencyMatrix, Matrix nodeFeatures, long embeddingDims) {
		super(model);
		this.adjacencyMatrix = adjacencyMatrix;
		constant("W", adjacencyMatrix);
		currentLayer = 0;
		if(embeddingDims!=0 && nodeFeatures!=null) {
			Matrix H0 = (Matrix) new DenseMatrix(adjacencyMatrix.getRows(), embeddingDims).setToRandom().setToNormalized();
			param("H0", H0);
			param("features", nodeFeatures);
			currentLayerInputDims = nodeFeatures.getCols()+embeddingDims;
			operation("H1 = features | H0");
			currentLayer = 1;
		}
		else if(nodeFeatures!=null) {
			param("H0", nodeFeatures);
			currentLayerInputDims = nodeFeatures.getCols();
		}
		else if(embeddingDims!=0) {
			Matrix H0 = (Matrix) new DenseMatrix(adjacencyMatrix.getRows(), embeddingDims).setToRandom().setToNormalized();
			param("H0", H0);
			currentLayerInputDims = embeddingDims;
		}
		else 
			throw new IllegalArgumentException();
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
	
	protected Matrix init(long rows, long cols) {
		//return (Matrix)(new DenseMatrix(rows, cols).setToRandom(new Normal(0, 1./Math.sqrt(cols))));
		double limit = Math.sqrt(6./(rows+cols));
		return (Matrix)(new DenseMatrix(rows, cols).setToRandom(new Uniform(-limit, limit)));
	}

	protected Tensor init(long size) {
		return new DenseTensor(size);//.setToRandom(new Normal(0, 1./Math.sqrt(size)));
	}
	
	public GCNBuilder addGCNLayer(String formula, long outputDims) {
		formula = formula.trim();
		if(formula.isEmpty())
			return this;
		if(formula.contains("\n")) {
			for(String subformula : formula.split("\\;|\\\n"))
				addGCNLayer(subformula, outputDims);
			return this;
		}
		if(formula.contains("B{l}"))
			param("B"+currentLayer, init(currentLayerInputDims, outputDims));
		if(formula.contains("W{l}"))
			param("W"+currentLayer, init(currentLayerInputDims, outputDims));
		if(formula.contains("R{l}"))
			param("R"+currentLayer, init(currentLayerInputDims, outputDims));
		if(formula.contains("b{l}"))
			param("b"+currentLayer, init(outputDims));
		layerFromFormula(formula, outputDims);
		return this;
	}

	/*public GCNBuilder transform(String activation, long outputDims, String layerName) {
		if(get("Wd"+layerName)==null)
			param("Wd"+layerName, init(currentLayerInputDims, outputDims));
		if(get("b"+layerName)==null)
			param("b"+layerName, init(outputDims));
		if(activation.equals("linear"))
			layerFromFormula("H{l+1} = H{l}@Wd{shared} + b{shared}".replace("{shared}", layerName), outputDims);
		else if(activation.equals("lrelu"))
			layerFromFormula("H{l+1} = prelu(H{l}@H{l}@Wd{shared} + b{shared}, 0.2 )".replace("{shared}", layerName), outputDims);
		else
			layerFromFormula("H{l+1} = "+activation+"(H{l}@H{l}@Wd{shared} + b{shared})".replace("{shared}", layerName), outputDims);
		return this;
	}*/
	
	public GCNBuilder multiclass(long numOutputs) {
		var("u");
		addGCNLayer("H{l+1} = W{l}@H{l}", numOutputs);
		layerOperation("prediction = max(H{l}[u])");
		out("prediction");
		return this;
	}
	
	public GCNBuilder similarity(String method) {
		if(method.equals("prediction")) {
			var("u");
			layerOperation("prediction = sigmoid(sum(H{l}[u]))");
			out("prediction");
		}
		else if(method.equals("distmult")) {
			var("u");
			var("v");
			param("DistMult", new DenseTensor(currentLayerInputDims).setToRandom().setToNormalized());
			layerOperation("sim = sigmoid(sum(H{l}[u]*H{l}[v]*DistMult))");
			out("sim");
		}
		else if(method.equals("dot")) {
			var("u");
			var("v");
			layerOperation("sim = sigmoid(sum(H{l}[u]*H{l}[v]))");
			out("sim");
		}
		else if(method.equals("disteuclidean")) {
			var("u");
			var("v");
			param("dimWeights", new DenseTensor(currentLayerInputDims).setToRandom().setToNormalized());
			layerOperation("diff = H{l}[u]-H{l}[v]");
			layerOperation("sim = sigmoid(-sum(diff*diff*dimWeights))");
			out("sim");
		}
		else if(method.equals("euclidean")) {
			var("u");
			var("v");
			layerOperation("diff = H{l}[u]-H{l}[v]");
			layerOperation("sim = sigmoid(-sum(diff*diff))");
			out("sim");
		}
		else if(method.equals("transe")) {
			var("u");
			var("v");
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
