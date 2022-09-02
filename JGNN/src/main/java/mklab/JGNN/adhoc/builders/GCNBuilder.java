package mklab.JGNN.adhoc.builders;

import java.util.HashMap;
import java.util.function.Function;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.ModelBuilder;

public class GCNBuilder extends ModelBuilder {
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
	public GCNBuilder futureConfigs(String config, Function<Integer, Double> func, int depth) {
		for(int layer=this.layer;layer<this.layer+depth;layer++) {
			String expression = config.replace("{l}", ""+layer);
			config(expression, func.apply(layer-this.layer));
		}
		return this;
	}
	public GCNBuilder futureConstants(String constantName, Function<Integer, Double> func, int depth) {
		for(int layer=this.layer;layer<this.layer+depth;layer++) {
			String expression = constantName.replace("{l}", ""+layer);
			constant(expression, func.apply(layer-this.layer));
		}
		return this;
	}
}
