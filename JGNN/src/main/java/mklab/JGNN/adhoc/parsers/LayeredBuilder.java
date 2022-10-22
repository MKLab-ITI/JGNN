package mklab.JGNN.adhoc.parsers;

import java.util.HashMap;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.adhoc.ModelBuilder;

public class LayeredBuilder extends ModelBuilder {
	private int layer = 0;
	private HashMap<String, Integer> rememberAs = new HashMap<String, Integer>();
	public LayeredBuilder() {
		this("h0");
	}
	public LayeredBuilder(String inputName) {
		var(inputName);
	}
	public LayeredBuilder var(String inputName) {
		super.var(inputName);
		return this;
	}
	public LayeredBuilder rememberAs(String layerId) {
		rememberAs.put(layerId, layer);
		return this;
	}
	public LayeredBuilder layer(String expression) {
		expression = expression
				.replace("{l+1}", ""+(layer+1))
			    .replace("{l}", ""+layer);
		for(String layerId : rememberAs.keySet())
			expression = expression.replace("{"+layerId+"}", ""+rememberAs.get(layerId));
		layer += 1;
		return operation(expression);
	}
	public LayeredBuilder classify() {
		layer("h{l+1}=h{l}[nodes]");
		layer("h{l+1}=softmax(h{l}, row)");
		out("h"+layer);
		return this;
	}
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
}
