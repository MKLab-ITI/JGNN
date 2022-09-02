package mklab.JGNN.nn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.nn.activations.LRelu;
import mklab.JGNN.nn.activations.PRelu;
import mklab.JGNN.nn.activations.Relu;
import mklab.JGNN.nn.activations.Sigmoid;
import mklab.JGNN.nn.activations.Tanh;
import mklab.JGNN.nn.inputs.Constant;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.inputs.Variable;
import mklab.JGNN.nn.operations.Add;
import mklab.JGNN.nn.operations.Complement;
import mklab.JGNN.nn.operations.Concat;
import mklab.JGNN.nn.operations.Dropout;
import mklab.JGNN.nn.operations.Gather;
import mklab.JGNN.nn.operations.Log;
import mklab.JGNN.nn.operations.MatMul;
import mklab.JGNN.nn.operations.Multiply;
import mklab.JGNN.nn.operations.Repeat;
import mklab.JGNN.nn.operations.Transpose;
import mklab.JGNN.nn.pooling.SoftMax;
import mklab.JGNN.nn.pooling.Sum;
import mklab.JGNN.nn.pooling.Max;

/**
 * This class and subclasses can be used to create {@link Model} instances 
 * by automatically creating and managing {@link NNOperation} instances based on
 * textual descriptions.
 * 
 * @author Emmanouil Krasanakis
 */
public class ModelBuilder {
	private String routing = "";
	private Model model = null;
	private HashMap<String, NNOperation> components = new HashMap<String, NNOperation>();
	private HashMap<String, Double> configurations = new HashMap<String, Double>();
	private int tmpVariableIdentifier = 0;
	public ModelBuilder() {
		this(new Model());
	}
	public ModelBuilder(Model model) {
		this.model = model;
	}
	public Model getModel() {
		return model;
	}
	/**
	 * Checks whether the builder has added to its managed model a component of 
	 * the given name.
	 * @param name The component name to check for.
	 * @return a <code>boolean</code> value
	 */
	public boolean hasComponent(String name) {
		return components.containsKey(name);
	}
	protected void assertValidName(String name) {
		if(name==null || name.isEmpty())
			throw new IllegalArgumentException("Invalid component name");
		if(configurations.containsKey(name)) 
			throw new IllegalArgumentException("Component name "+name+" already in use as a configuration");
		if(components.containsKey(name)) 
			throw new IllegalArgumentException("Component name "+name+" already in use by another model component");
	}
	protected void assertExists(String name) {
		if(configurations.containsKey(name))
			throw new IllegalArgumentException("Component name "+name+" is a configuration but expressions can only parse components");
		if(!components.containsKey(name))
			throw new IllegalArgumentException("Component name "+name+" not declared");
	}
	
	public ModelBuilder var(String name) {
		assertValidName(name);
		Variable variable = new Variable();
		components.put(name, variable);
		model.addInput(variable);
		variable.setDescription(name);
		return this;
	}
	
	/**
	 * Declares the component with the given name an output of the 
	 * managed model. The component should have already been assigned a value.
	 * To output complex expressions use {@link #operation(String)}
	 * to define them first.
	 * @param name A component name.
	 * @return The builder's instance.
	 */
	public ModelBuilder out(String name) {
		assertExists(name);
		model.addOutput(components.get(name));
		return this;
	}
	
	/**
	 * Declares a learnable {@link Paramater} component with the given name,
	 * learning L2 regularization, and initial value.
	 * @param name The name to be assigned to the new component.
	 * @param regularization The regularization value. Zero corresponds to no regularization.
	 * 	Typically, this is non-negative.
	 * @param value The initial value to be assigned to the parameter. Exact values
	 * 	can be overridden by neural initialization strategies, but an initial value
	 *  should be declared nonetheless to determine the parameter type and allocate
	 *  any necessary memory.
	 * @return The builder's instance.
	 * @see #param(String, Tensor)
	 * @see #operation(String)
	 */
	public ModelBuilder param(String name, double regularization, Tensor value) {
		assertValidName(name);
		NNOperation variable = new Parameter(value, regularization);
		components.put(name, variable);
		variable.setDescription(name);
		return this;
	}
	
	/**
	 * Declares a configuration hyperparameter, which can be used to declare
	 * matrix and vector parameters during {@link #operation(String)} expressions.
	 * For in-expression use of hyperparameters, delcare them with {@link #constant(String, double)}.
	 * @param name The name of the configuration hyperparameter.
	 * @param value The value to be assigned to the hyperparameter.
	 * 	Typically, provide a long number.
	 * @return The builder's instance.
	 * @see #operation(String)
	 * @see #param(String, Tensor)
	 * @see #param(String, double, Tensor)
	 */
	public ModelBuilder config(String name, double value) {
		configurations.put(name, value);
		return this;
	}
	
	protected double parseConfigValue(String text) {
		if(configurations.containsKey(text))
			return configurations.get(text);
		return Double.parseDouble(text);
	}
	
	protected boolean isDouble(String text) {
		try {
			Double.parseDouble(text);
			return true;
		}
		catch(Exception e) {
			return false;
		}
	}

	/**
	 * Declares a learnable {@link Paramater} component with the given name,
	 * zero regularization, and initial value.
	 * @param name The name to be assigned to the new component.
	 * @param value The initial value to be assigned to the parameter. Exact values
	 * 	can be overridden by neural initialization strategies, but an initial value
	 *  should be declared nonetheless to determine the parameter type and allocate
	 *  any necessary memory.
	 * @return The builder's instance.
	 * @see #param(String, double, Tensor)
	 * @see #operation(String)
	 */
	public ModelBuilder param(String name, Tensor value) {
		assertValidName(name);
		NNOperation variable = new Parameter(value);
		components.put(name, variable);
		variable.setDescription(name);
		return this;
	}
	
	public ModelBuilder constant(String name, double value) {
		return constant(name, Tensor.fromDouble(value));
	}
	
	public ModelBuilder constant(String name, Tensor value) {
		if(components.containsKey(name)) {
			((Constant)components.get(name)).set(value);
			((Constant)components.get(name)).setDescription(name);
			return this;
		}
		assertValidName(name);
		NNOperation variable = new Constant(value);
		components.put(name, variable);
		variable.setDescription(name);
		return this;
	}
	
	public NNOperation get(String name) {
		return components.get(name);
	}
	
	public ModelBuilder runModel(Tensor... inputs) {
		model.predict(inputs);
		return this;
	}
	
	public ModelBuilder runModel(ArrayList<Tensor> inputs) {
		model.predict(inputs);
		return this;
	}
	
	public ModelBuilder operation(String desc) {
		
		//System.out.println(desc);
		
		String[] lines = desc.split("\\;|\\\n");
		if(lines.length>1) {
			for(String line : lines)
				if(!line.trim().isEmpty())
					operation(line);
			return this;
		}
		
		desc = desc.trim();
		desc = desc.replace("=", " = ");
		desc = desc.replace("@", " @ ");
		desc = desc.replace("+", " + ");
		desc = desc.replace("*", " * ");
		desc = desc.replace("[", " [ ");
		desc = desc.replace("]", " ] ");
		desc = desc.replace("(", " ( ");
		desc = desc.replace(")", " ) ");
		desc = desc.replace(",", " , ");
		if(!desc.contains("MINUS_ONE")) {
			if(desc.contains("-") && !components.containsKey("MINUS_ONE"))
				constant("MINUS_ONE", Tensor.fromDouble(-1));
			desc = desc.replace("-", " + MINUS_ONE * ");
		}
		desc = desc.replaceAll("\\s\\=\\s+\\+\\s+MINUS\\_ONE", " = MINUS_ONE");
		desc = desc.replaceAll("\\s+", " ");
		
		boolean madeChanges = true;
		while(madeChanges) {
			madeChanges = false;
			String newDesc = "";
			ArrayList<StringBuilder> suboperation = new ArrayList<StringBuilder>();
			suboperation.add(new StringBuilder());
			int level = 0;
			for(int i=0;i<desc.length();i++) {
				char c = desc.charAt(i);
				if(c=='(') {
					if(level!=0)
						suboperation.get(suboperation.size()-1).append(c);
					level += 1;
				}
				else if(c==',' && level==1) {
					suboperation.add(new StringBuilder());
				}
				else if(c==')') {
					level -= 1;
					if(level!=0)
						suboperation.get(suboperation.size()-1).append(c);
					else {
						String args = "";
						for(StringBuilder subop : suboperation) {
							if(!args.isEmpty())
								args += " , ";
							String arg = subop.toString().trim();
							if(components.containsKey(arg)
									|| configurations.containsKey(arg) 
									|| (newDesc.endsWith(" matrix ") && isDouble(arg)) 
									|| (newDesc.endsWith(" vector ") && isDouble(arg)) 
									|| arg.equals("col") || arg.equals("row")) 
								args += arg;
							else {
								String tmpName = "_tmp"+tmpVariableIdentifier;
								tmpVariableIdentifier += 1;
								while(components.containsKey(tmpName)) {
									tmpName = "_tmp"+tmpVariableIdentifier;
									tmpVariableIdentifier += 1;
								}
								operation(tmpName+" = "+arg);
								args += tmpName;
								madeChanges = true;
							}
						}
						suboperation.clear();
						suboperation.add(new StringBuilder());
						if(!components.containsKey(args) && !configurations.containsKey(args))
							newDesc += "( "+args+" )";
						else
							newDesc += args;
					}
				}
				else if(level>0)
					suboperation.get(suboperation.size()-1).append(c);
				else
					newDesc += c;
			}
			if(level!=0)
				throw new RuntimeException("Imbalanced parenthesis in operation: "+desc);
			desc = newDesc;
			String[] operators = {" + ", " * ", " @ ", " | "};
			madeChanges = false;
			for(String operator : operators) {
				if(madeChanges)
					break;
				String[] splt = desc.split("\\s*=\\s*");
				if(splt.length!=2)
					throw new IllegalArgumentException("Exactly one equality needed in each operation: "+desc);
				newDesc = "";
				int lastArgPos = -1;
				for(int i=0;i<splt[1].length();i++) {
					char c = splt[1].charAt(i);
					if(c=='(') {
						level += 1;
						newDesc += c;
					}
					else if(c==')') {
						level -= 1;
						newDesc += c;
					}
					else {
						newDesc += c;
						if(!madeChanges && level==0 && newDesc.endsWith(operator)) {
							String arg = newDesc.substring(0, newDesc.length()-operator.length()).trim();
							if(arg.startsWith("(") && arg.endsWith(")"))
								arg = arg.substring(1, arg.length()-1).trim();
							if(components.containsKey(arg) || configurations.containsKey(arg)) 
								newDesc = arg+operator;
							else {
								newDesc = "("+arg+")"+operator;
								madeChanges = true;
							}
						}
						if(level==0 && newDesc.endsWith(operator))
							lastArgPos = newDesc.length();
					}
				}
				if(lastArgPos!=-1) {
					String arg = newDesc.substring(lastArgPos).trim();
					newDesc = newDesc.substring(0, lastArgPos);
					if(arg.startsWith("(") && arg.endsWith(")"))
						arg = arg.substring(1, arg.length()-1).trim();
					if(components.containsKey(arg) || configurations.containsKey(arg)) 
						newDesc += arg;
					else {
						newDesc += "("+arg+")";
						madeChanges = true;
					}
				}
				desc = splt[0]+" = "+newDesc;
			}
		}
		
		String prevRouting = routing;
		routing += desc + "\n";
		desc = desc.replace("(", "").replace(")", "").replace(",", "")+" ";
		
		
		String[] splt = desc.split("\\s+");
		String name = splt[0];
		String arg0 = null;
		String arg1 = null;
		assertValidName(name);
		NNOperation component;
		if(splt.length==3) {
			try {
				double val = Double.parseDouble(splt[2]);
				constant(name, Tensor.fromDouble(val));
			}
			catch(NumberFormatException e) {
				throw new RuntimeException("Symbol "+splt[2]+" not defined.");
			}
			return this;
		}
		else if(splt[3].equals("+")) {
			component = new Add();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[2].equals("softmax")) {
			boolean mode = false;
			if(splt.length>4) {
				String modeText = splt[4].trim();
				if(modeText.equals("col"))
					mode = false;
				else if(modeText.equals("row"))
					mode = true;
				else
					throw new RuntimeException("Invalid argument "+modeText+" to softmax");
			}
			component = new SoftMax(mode);
			arg0 = splt[3];
		}
		else if(splt[2].equals("sum")) {
			boolean mode = false;
			if(splt.length>4) {
				String modeText = splt[4].trim();
				if(modeText.equals("col"))
					mode = false;
				else if(modeText.equals("row"))
					mode = true;
				else
					throw new RuntimeException("Invalid argument "+modeText+" to softmax");
			}
			component = new Sum(mode);
			arg0 = splt[3];
		}
		else if(splt[2].equals("max")) {
			boolean mode = false;
			if(splt.length>4) {
				String modeText = splt[4].trim();
				if(modeText.equals("col"))
					mode = false;
				else if(modeText.equals("row"))
					mode = true;
				else
					throw new RuntimeException("Invalid argument "+modeText+" to softmax");
			}
			component = new Max(mode);
			arg0 = splt[3];
		}
		else if(splt[2].equals("matrix") || splt[2].equals("mat")) {
			param(name, splt.length>5?parseConfigValue(splt[5]):0.,
					new DenseMatrix((long)parseConfigValue(splt[3]), (long)parseConfigValue(splt[4]))
					.setDimensionName(isDouble(splt[3])?null:splt[3], isDouble(splt[4])?null:splt[4]));
			routing = prevRouting;
			return this;
		}
		else if(splt[2].equals("vector") || splt[2].equals("vec")) {
			param(name, 
					splt.length>4?parseConfigValue(splt[4]):0.,
					new DenseTensor((long)parseConfigValue(splt[3]))
					.setDimensionName(isDouble(splt[3])?null:splt[3]));
			routing = prevRouting;
			return this;
		}
		else if(splt[2].equals("relu")) {
			component = new Relu();
			arg0 = splt[3];
		}
		else if(splt[2].equals("lrelu")) {
			component = new LRelu();
			arg0 = splt[3];
			arg1 = splt[4];
		}
		else if(splt[2].equals("prelu")) {
			component = new PRelu();
			arg0 = splt[3];
			arg1 = splt[4];
		}
		else if(splt[2].equals("tanh")) {
			component = new Tanh();
			arg0 = splt[3];
		}
		else if(splt[2].equals("log")) {
			component = new Log();
			arg0 = splt[3];
		}
		else if(splt[2].equals("debug")) {
			component = this.get(splt[3]);
			component.debugging = true;
		}
		else if(splt[2].equals("transpose")) {
			component = new Transpose();
			arg0 = splt[3];
		}
		else if(splt[2].equals("sigmoid")) {
			component = new Sigmoid();
			arg0 = splt[3];
		}
		else if(splt[2].equals("complement")) {
			component = new Complement();
			arg0 = splt[3];
		}
		else if(splt[2].equals("repeat")) {
			component = new Repeat();
			arg0 = splt[3];
			arg1 = splt[4];
		}
		else if(splt[2].equals("dropout") || splt[2].equals("drop")) {
			component = new Dropout();
			arg0 = splt[3];
			arg1 = splt[4];
		}
		else if(splt[2].equals("transpose")) {
			component = new Transpose();
			arg0 = splt[3];
		}
		else if(splt[3].equals("|")) {
			component = new Concat();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[3].equals("*")) {
			component = new Multiply();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[3].equals("@")) {
			component = new MatMul();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[3].equals("[")) {
			component = new Gather();
			arg0 = splt[4];
			arg1 = splt[2];
		}
		else if(splt[3].equals("x")) {
			component = new Repeat();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else
			throw new RuntimeException("Invalid operation: "+desc);

		if(arg0!=null) {
			assertExists(arg0);
			component.addInput(components.get(arg0));
		}
		if(arg1!=null) {
			assertExists(arg1);
			component.addInput(components.get(arg1));
		}
		
		components.put(name, component);
		component.setDescription(name);
		
		return this;
	}

	public ModelBuilder assertForwardValidity(List<Integer> inputSizes) {
		ArrayList<Tensor> inputs = new ArrayList<Tensor>();
		for(int size : inputSizes)
			inputs.add(new DenseTensor(size));
		model.predict(inputs);
		return this;
	}
	
	public ModelBuilder assertBackwardValidity() {
		HashSet<NNOperation> allFoundComponents = new HashSet<NNOperation>();
		Stack<NNOperation> pending = new Stack<NNOperation>();
		for(NNOperation output : model.getOutputs()) {
			pending.add(output);
			allFoundComponents.add(output);
		}
		while(!pending.isEmpty()) {
			NNOperation component = pending.pop();
			for(NNOperation componentInput : component.getInputs()) 
				if(!allFoundComponents.contains(componentInput)) {
					allFoundComponents.add(componentInput);
					pending.add(componentInput);
				}
		}

		HashSet<NNOperation> actualComponents = new HashSet<NNOperation>(components.values());
		for(NNOperation component : allFoundComponents)
			if(!actualComponents.contains(component))
				System.err.println("The component "+component.describe()+" was not added by this builder to its model's pipeline");
		for(NNOperation component : actualComponents)
			if(!allFoundComponents.contains(component)) {
				throw new RuntimeException("The component "+component.describe()+" does not lead to an output");
				//System.err.println("The component "+component.describe()+" does not lead to an output and will be removed from the outputs of other components");
				//for(NNOperation other : actualComponents)
				//	other.getOutputs().remove(component);
			}
		
		return this;
	}
	
	public String describe() {
		getModel();
		return routing;
	}
	public String getExecutionGraphDot() {
		getModel();
		String ret = "//Can visualize at: https://dreampuf.github.io/GraphvizOnline";
		ret+="\ndigraph operations {";
		for(NNOperation component : components.values()) {
			for(NNOperation input : component.getInputs())
				ret+="\n   "+input.getDescription()+" -> "+component.getDescription();
		}
		for(NNOperation component : components.values()) 
			if(model.getOutputs().contains(component)) {
				ret+="\n   "+component.getDescription()
					+"[label=\""+component.getDescription()+" = "+component.getClass().getSimpleName()+"\", shape=doubleoctagon]";
			}
			else if(component instanceof Variable)
				ret+="\n   "+component.getDescription()+"[color=red,shape=octagon]";
			else if(component instanceof Constant) {
				if(((Constant)component).get().size()==1) {
					if(component.getDescription().startsWith("_"))
						ret+="\n   "+component.getDescription()
						+"[shape=rectangle,color=red,label=\""
							+((Parameter)component).get().toDouble()+"\"]";
					else
						ret+="\n   "+component.getDescription()
						+"[shape=rectangle,color=red,label=\""
							+component.getDescription()+" = "+((Parameter)component).get().toDouble()+"\"]";
				}
				else if(component.getDescription().startsWith("_")){
					ret+="\n   "+component.getDescription()
					+"[shape=rectangle,color=red,label=\""
							+((Parameter)component).get().describe()+"\"]";
				}
				else
					ret+="\n   "+component.getDescription()
					+"[shape=rectangle,color=red,label=\""
						+component.getDescription()+" = "+((Parameter)component).get().describe()+"\"]";
			
			}
			else if(component instanceof Parameter) {
				//ret+="\n   "+component.getDescription()+"[color=green]";
				if(component.getDescription().startsWith("_")){
					ret+="\n   "+component.getDescription()
					+"[shape=rectangle,color=green,label=\""
							+((Parameter)component).get().describe()+"\"]";
				}
				else
					ret+="\n   "+component.getDescription()
					+"[shape=rectangle,color=green,label=\""
						+component.getDescription()+" = "+((Parameter)component).get().describe()+"\"]";
			
			}
			else if(component.getDescription().startsWith("_")){
				ret+="\n   "+component.getDescription()+"[label="+component.getClass().getSimpleName()+"]";
			}
			else
				ret+="\n   "+component.getDescription()+"[label=\""+component.getDescription()+" = "+component.getClass().getSimpleName()+"\"]";
			
		ret += "\n}";
		return ret;
	}
	public ModelBuilder print() {
		getModel();
		for(NNOperation component : components.values())
			if(component instanceof Parameter)
				System.out.println(component.describe());
		System.out.println(describe());
		return this;
	}
	public ModelBuilder printState() {
		getModel();
		for(NNOperation component : components.values())
			System.out.println(component.view());
		return this;
	}
}
