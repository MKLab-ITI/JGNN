package mklab.JGNN.adhoc.parsers;

import java.util.HashMap;
import java.util.function.Function;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.adhoc.ModelBuilder;

/**
 * Extends the capabilities of the {@link ModelBuilder}
 * with the ability to define multilayer (e.g. deep) neural architectures.
 * The symbols <code>{l}</code> and <code>{l+1}</code> are replaced in all expressions
 * with appropriate layer identifiers (these increase by one each time a new
 * {@link #layer(String)} is defined.
 * 
 * @see #layer(String)
 * @see #layerRepeat(String, int)
 * @see #futureConfigs(String, Function, int)
 * @see #futureConstants(String, Function, int)
 * @see #rememberAs(String)
 */
public class LayeredBuilder extends ModelBuilder {
	private int layer = 0;
	private HashMap<String, Integer> rememberAs = new HashMap<String, Integer>();
	/**
	 * Instantiates a layered builder with input name <code>h0</code>. This can be 
	 * used by future expressions involving <code>h{l}</code>. You can add more
	 * architecture inputs normally with {@link #var(String)}.
	 * 
	 * @see #LayeredBuilder(String)
	 */
	public LayeredBuilder() {
		this("h0");
	}
	/**
	 * Instantiates a layered builder with the given symbol as an input name.
	 * If you plan to immediately use a {@link #layer(String)} expression
	 * that involves <code>X{l}</code>, where <code>X</code> is some symbol,
	 * set <code>X0</code> as the architecture's input. You can add more
	 * architecture inputs normally with {@link #var(String)}.
	 * 
	 * @param inputName The symbol to use as the built architecture's input.
	 */
	public LayeredBuilder(String inputName) {
		var(inputName);
	}
	public LayeredBuilder var(String inputName) {
		super.var(inputName);
		return this;
	}
	/**
	 * Sets the current layer identifier to a specific symbol <code>layerId</code>
	 * so that future usage of <code>{layerId}</code> is automatically replaced with 
	 * the identifier.
	 * @param layerId The symbol to set to the current layer identifier.
	 * @return <code>this</code> layer builder.
	 */
	public LayeredBuilder rememberAs(String layerId) {
		rememberAs.put(layerId, layer);
		return this;
	}
	/**
	 * Applies an {@link #operation(String)} and increases the layer identifier count.
	 * @param expression A parsable expression.
	 * @return <code>this</code> layer builder.
	 * @see #layerRepeat(String, int)
	 */
	public LayeredBuilder layer(String expression) {
		expression = expression
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			expression = expression.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
		layer += 1;
		return operation(expression);
	}
	/*public LayeredBuilder classify() {
		layer("h{l+1}=h{l}[nodes]");
		layer("h{l+1}=softmax(h{l}, row)");
		out("h"+layer);
		return this;
	}*/
	/**
	 * Repeats a {@link #layer(String)} definition a number of times.
	 * Ideal for building deep architectures.
	 * @param expression The expression to repeat for each layer.
	 * @param times The number of times to repeat the expression.
	 * @return <code>this</code> layer builder.
	 * 
	 * @see #futureConfigs(String, Function, int)
	 * @see #futureConstants(String, Function, int)
	 */
	public LayeredBuilder layerRepeat(String expression, int times) {
		for(int i=0;i<times;i++)
			layer(expression);
		return this;
	}
	public LayeredBuilder config(String name, double value) {
		super.config(name, value);
		return this;
	}
	public LayeredBuilder param(String name, Tensor value) {
		super.param(name, value);
		return this;
	}
	public LayeredBuilder constant(String name, double value) {
		super.constant(name, value);
		return this;
	}
	public LayeredBuilder constant(String name, Tensor value) {
		super.constant(name, value);
		return this;
	}
	public LayeredBuilder param(String name, double regularization, Tensor value) {
		super.param(name, regularization, value);
		return this;
	}
	public LayeredBuilder operation(String desc) {
		desc = desc
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			desc = desc.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
		super.operation(desc);
		return this;
	}
	public LayeredBuilder out(String expression) {
		expression = expression
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			expression = expression.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
		super.out(expression);
		return this;
	}
	/**
	 * Concatenates horizontally the output of a number of given layers,
	 * starting from the last one and going backwards. (For concatenation
	 * of specific layers just use <code>concat</code> within normal operations.)
	 * @param depth The number of given layers to concatenate.
	 * @return <code>this</code> layer builder.
	 */
	public LayeredBuilder concat(int depth) {
		String expression = "";
		for(int i=layer;i>layer-depth;i--) {
			if(!expression.isEmpty())
				expression += " | ";
			expression += "h"+i;
		}
		layer("h{l+1} = "+expression);
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
	 * @return <code>this</code> layer builder.
	 * 
	 * @see #futureConstants(String, Function, int)
	 */
	public LayeredBuilder futureConfigs(String config, Function<Integer, Double> func, int depth) {
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
	 * @return <code>this</code> layer builder.
	 * 
	 * @see #futureConstants(String, Function, int)
	 */
	public LayeredBuilder futureConstants(String constantName, Function<Integer, Double> func, int depth) {
		for(int layer=this.layer;layer<this.layer+depth;layer++) {
			String expression = constantName.replace("{l}", ""+layer);
			constant(expression, func.apply(layer-this.layer));
		}
		return this;
	}
}
