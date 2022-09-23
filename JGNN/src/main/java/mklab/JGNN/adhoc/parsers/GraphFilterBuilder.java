package mklab.JGNN.adhoc.parsers;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.Model;
import mklab.JGNN.adhoc.ModelBuilder;


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
	
	public GraphFilterBuilder set(long dims) {
		if(layer>0)
			throw new RuntimeException("Can only add embedding layers before adding normal layers");
		operation("encoded = relu("+labelMatrix+" @ matrix("+labels.getCols()+","+dims+") + vector("+dims+"))");
		embeddingDims = dims;
		labelMatrix = "encoded";
		return this;
	}
	public GraphFilterBuilder addShift(double weight) {
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
			this.var("nodes");
			operation("result = sum"+layer+"[nodes]");
			this.out("result");
		}
		return super.getModel();
	}

}