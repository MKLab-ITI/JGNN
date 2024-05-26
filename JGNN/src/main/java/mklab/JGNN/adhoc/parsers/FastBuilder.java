package mklab.JGNN.adhoc.parsers;

import java.util.HashMap;
import java.util.function.Function;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.adhoc.ModelBuilder;

/**
 * Extends the capabilities of {@link LayeredBuilder} to use
 * for node classification. It accepts the adjacency graph in the constructor,
 * to be used with the symbol <i>A</i> in operations or layer definitions,
 * and node features.
 * 
 * @author Emmanouil Krasanakis
 * @see #classify()
 */
public class FastBuilder extends ModelBuilder {
	private int layer = 0;
	private HashMap<String, Integer> rememberAs = new HashMap<String, Integer>();
	/**
	 * @deprecated This constructor should only be used by loading.
	 */
	public FastBuilder() {
	}
	/**
	 * Creates a graph neural network builder from an 
	 * normalized adjacency matrix and a node feature matrix.
	 * @param adjacency The pre-normalized adjacency matrix.
	 * @param features The node feature matrix.
	 */
	public FastBuilder(Matrix adjacency, Matrix features) {
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
	public FastBuilder rememberAs(String layerId) {
		rememberAs.put(layerId, layer);
		return this;
	}
	/**
	 * Applies an {@link #operation(String)} and increases the layer identifier count.
	 * @param expression A parsable expression.
	 * @return <code>this</code> builder.
	 * @see #layerRepeat(String, int)
	 */
	public FastBuilder layer(String expression) {
		expression = expression
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			expression = expression.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
		layer += 1;
		super.operation(expression);
		return this;
	}
	/**
	 * Adds a classification layer that gather the number of inputs nodes
	 * and applies softmax on all of them.
	 * @return <code>this</code> builder.
	 */
	public FastBuilder classify() {
		var("nodes");
		layer("h{l+1}=h{l}[nodes]");
		layer("h{l+1}=softmax(h{l}, row)");
		out("h"+layer);
		return this;
	}
	/**
	 * Repeats a {@link #layer(String)} definition a number of times.
	 * Ideal for building deep architectures.
	 * @param expression The expression to repeat for each layer.
	 * @param times The number of times to repeat the expression.
	 * @return <code>this</code> builder.
	 * 
	 * @see #futureConfigs(String, Function, int)
	 * @see #futureConstants(String, Function, int)
	 */
	public FastBuilder layerRepeat(String expression, int times) {
		for(int i=0;i<times;i++)
			layer(expression);
		return this;
	}
	public FastBuilder function(String name, String value) {
		super.function(name, value);
		return this;
	}
	public FastBuilder config(String name, double value) {
		super.config(name, value);
		return this;
	}
	public FastBuilder param(String name, Tensor value) {
		super.param(name, value);
		return this;
	}
	public FastBuilder constant(String name, double value) {
		super.constant(name, value);
		return this;
	}
	public FastBuilder constant(String name, Tensor value) {
		super.constant(name, value);
		return this;
	}
	public FastBuilder param(String name, double regularization, Tensor value) {
		super.param(name, regularization, value);
		return this;
	}
	public FastBuilder operation(String desc) {
		desc = desc
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			desc = desc.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
		super.operation(desc);
		return this;
	}
	/**
	 * Defines a number of {@link #config(String, double)} symbols involving a <code>{l}</code>
	 * notation, for example so that they can be used during {@link #layerRepeat(String, int)}. 
	 * @param config The configuration symbols (these should involve <code>{l}</code>).
	 * @param func A lambda Java function to calculate the configuration's value. This takes
	 *  as input an integer (starting from 0 for the current layer) and adds one for each
	 *  subsequently declared symbol.
	 * @param depth The number of future layers expected to use the symbols.
	 * @return <code>this</code> builder.
	 * 
	 * @see #futureConstants(String, Function, int)
	 */
	public FastBuilder futureConfigs(String config, Function<Integer, Double> func, int depth) {
		for(int layer=this.layer;layer<this.layer+depth;layer++) {
			String expression = config.replace("{l}", ""+layer);
			config(expression, func.apply(layer-this.layer));
		}
		return this;
	}
	/**
	 * Defines a number of {@link #constant(String, double)} symbols involving a <code>{l}<code>
	 * notation, for example so that they can be used during {@link #layerRepeat(String, int)}. 
	 * @param constantName The configuration symbols (these should involve <code>{l}</code>).
	 * @param func A lambda Java function to calculate the constant's value. This takes
	 *  as input an integer (starting from 0 for the current layer) and adds one for each
	 *  subsequently declared symbol.
	 * @param depth The number of future layers expected to use the constant.
	 * @return <code>this</code> builder.
	 * 
	 * @see #futureConstants(String, Function, int)
	 */
	public FastBuilder futureConstants(String constantName, Function<Integer, Double> func, int depth) {
		for(int layer=this.layer;layer<this.layer+depth;layer++) {
			String expression = constantName.replace("{l}", ""+layer);
			constant(expression, func.apply(layer-this.layer));
		}
		return this;
	}
	/**
	 * Concatenates horizontally the output of a number of given layers,
	 * starting from the last one and going backwards. (For concatenation
	 * of specific layers just use <code>concat</code> within normal operations.)
	 * @param depth The number of given layers to concatenate.
	 * @return <code>this</code> builder.
	 */
	public FastBuilder concat(int depth) {
		String expression = "";
		for(int i=layer;i>layer-depth;i--) {
			if(!expression.isEmpty())
				expression += " | ";
			expression += "h"+i;
		}
		layer("h{l+1} = "+expression);
		return this;
	}
}
