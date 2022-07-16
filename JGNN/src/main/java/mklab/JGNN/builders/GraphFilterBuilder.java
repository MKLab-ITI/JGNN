package mklab.JGNN.builders;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;

public class GraphFilterBuilder extends ModelBuilder {
	private int layer = 0;
	private String labelMatrix;
	private Matrix adjacency;
	private Matrix labels;
	private long embeddingDims = 0;
	public GraphFilterBuilder(Matrix adjacency, Matrix labels) {
		super();
		this.adjacency = adjacency;
		this.labels = labels;
		this.constant("adjacency", adjacency);
		this.constant("labels", labels);
		embeddingDims = labels.getCols();
		labelMatrix = "labels";
		operation("sum0 = 0");
	}
	public GraphFilterBuilder addEmbeddingLayer(long dims) {
		if(layer>0)
			throw new RuntimeException("Can only add embedding layers befor adding normal layers");
		operation("encoded = relu("+labelMatrix+" @ matrix("+labels.getCols()+","+dims+") + vector("+dims+"))");
		embeddingDims = dims;
		labelMatrix = "encoded";
		return this;
	}
	public GraphFilterBuilder addLayer(double weight) {
		constant("h"+layer, Tensor.fromDouble(weight));
		if(layer==0) {
			operation("sum1 = sum0 + h0*"+labelMatrix);
			operation("pow0 = "+labelMatrix+" + 0");
		}
		else{
			operation("pow"+layer+" = adjacency@pow"+(layer-1));
			operation("sum"+(layer+1)+" = sum"+layer+" + h"+layer+" * pow"+layer);
		}
		layer += 1;
		return this;
	}
	public Model getModel() {
		if(super.getModel().getOutputs().size()==0) {
			if(labelMatrix.equals("encoded")) {
				operation("decoded = sum"+layer+" @ matrix("+embeddingDims+","+labels.getCols()+")");
				this.var("nodes");
				operation("result = softmax(decoded[nodes],row)");
				this.out("result");
			}
			else {
				this.var("nodes");
				operation("result = sum"+layer+"[nodes]");
				this.out("result");
			}
		}
		return super.getModel();
	}

}
