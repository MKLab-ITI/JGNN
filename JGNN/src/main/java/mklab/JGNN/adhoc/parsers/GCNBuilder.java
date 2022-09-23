package mklab.JGNN.adhoc.parsers;

import java.util.HashMap;
import java.util.function.Function;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.adhoc.ModelBuilder;

/**
 * This extends the capabilities of {@link LayeredBuilder} to use
 * for node classification. It accepts the adjacency graph in the constructor,
 * to be used with the name <it>A</it> in operations or layer definitions,
 * and node features.
 * @author Emmanouil Krasanakis
 */
public class GCNBuilder extends ModelBuilder {
	private int layer = 0;
	private HashMap<String, Integer> rememberAs = new HashMap<String, Integer>();
	/**
	 * @deprecated This constructor should only be used by loading.
	 */
	public GCNBuilder() {
	}
	public GCNBuilder(Matrix adjacency, Matrix features) {
		long numFeatures = features.getCols();
		config("features", numFeatures);
		constant("A", adjacency);
		constant("h0", features);
	}
	protected String saveCommands() {
		String ret = super.saveCommands();
		for(String rememberKey : rememberAs.keySet())
			ret += "remember "+rememberKey+" as "+rememberAs.get(rememberKey)+"\n";
		ret += "layer "+layer+"\n";
		return ret;
	}
	protected boolean loadCommand(String command, String data) {
		if(command.equals("layer")) {
			layer = Integer.parseInt(data);
			return true;
		}
		if(command.equals("remember")) {
			int pos = data.lastIndexOf(" as ");
			rememberAs.put(data.substring(0, pos), Integer.parseInt(data.substring(pos+4)));
			return true;
		}
		return super.loadCommand(command, data);
	}
	/**
	 * Remembers the last layer's output per a given identifier so that {layerId}
	 * within future {@link #layer(String)} definitions is made to refer to the
	 * current layer.
	 * @param layerId An identifier to remember the last layer's output as.
	 * @return The model builder.
	 */
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
		super.operation(expression);
		return this;
	}
	public GCNBuilder classify() {
		var("nodes");
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
		desc = desc
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			desc = desc.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
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
