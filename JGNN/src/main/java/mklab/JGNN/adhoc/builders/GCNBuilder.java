package mklab.JGNN.adhoc.builders;

import java.util.HashMap;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;

public class GCNBuilder extends LayeredBuilder {
	private int layer = 0;
	private HashMap<String, Integer> rememberAs = new HashMap<String, Integer>();
	public GCNBuilder(Matrix adjacency, Matrix features) {
		long numFeatures = features.getCols();
		config("features", numFeatures);
		constant("A", adjacency);
		constant("h0", features);
		var("nodes");
	}
	public GCNBuilder rememberAs(String layerId) {
		rememberAs.put(layerId, layer);
		return this;
	}
	public GCNBuilder layer(String expression) {
		expression = expression
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			expression = expression.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
		layer += 1;
		return operation(expression);
	}
	public GCNBuilder classify() {
		layer("h{l+1}=h{l}[nodes]");
		layer("h{l+1}=softmax(h{l}, row)");
		out("h"+layer);
		return this;
	}
	public GCNBuilder layerRepeat(String expression, int times) {
		for(int i=0;i<times;i++)
			layer(expression);
		return this;
	}
	public GCNBuilder config(String name, double value) {
		super.config(name, value);
		return this;
	}
	public GCNBuilder param(String name, Tensor value) {
		super.param(name, value);
		return this;
	}
	public GCNBuilder constant(String name, double value) {
		super.constant(name, value);
		return this;
	}
	public GCNBuilder constant(String name, Tensor value) {
		super.constant(name, value);
		return this;
	}
	public GCNBuilder param(String name, double regularization, Tensor value) {
		super.param(name, regularization, value);
		return this;
	}
	public GCNBuilder operation(String desc) {
		super.operation(desc);
		return this;
	}
}
