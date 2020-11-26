package mklab.JGNN.core.operations;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

import mklab.JGNN.core.operations.NN.Add;
import mklab.JGNN.core.operations.NN.Constant;
import mklab.JGNN.core.operations.NN.LRelu;
import mklab.JGNN.core.operations.NN.Log;
import mklab.JGNN.core.operations.NN.MatMul;
import mklab.JGNN.core.operations.NN.Multiply;
import mklab.JGNN.core.operations.NN.PRelu;
import mklab.JGNN.core.operations.NN.Parameter;
import mklab.JGNN.core.operations.NN.Relu;
import mklab.JGNN.core.operations.NN.Repeat;
import mklab.JGNN.core.operations.NN.Sigmoid;
import mklab.JGNN.core.operations.NN.Gather;
import mklab.JGNN.core.operations.NN.Sum;
import mklab.JGNN.core.operations.NN.Tanh;
import mklab.JGNN.core.operations.NN.Variable;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.primitives.tensor.DenseTensor;

public class ModelBuilder {
	private String routing = "";
	private Model model = new Model();
	private HashMap<String, NNOperation> components = new HashMap<String, NNOperation>();
	private int tmpVariableIdentifier = 0;
	public ModelBuilder() {
	}
	public ModelBuilder(Model model) {
		this.model = model;
	}
	public Model getModel() {
		return model;
	}
	protected void assertValidName(String name) {
		if(name==null || name.isEmpty())
			throw new RuntimeException("Invalid component name");
		if(components.containsKey(name))
			throw new RuntimeException("Variable name "+name+" already in use by another model component");
	}
	protected void assertExists(String name) {
		if(!components.containsKey(name))
			throw new RuntimeException("Component name "+name+" not declared");
	}
	
	public ModelBuilder var(String name) {
		assertValidName(name);
		Variable variable = new Variable();
		components.put(name, variable);
		model.addInput(variable);
		variable.setDescription(name);
		return this;
	}

	public ModelBuilder out(String name) {
		assertExists(name);
		model.addOutput(components.get(name));
		return this;
	}
	
	public ModelBuilder param(String name, Tensor value) {
		assertValidName(name);
		NNOperation variable = new Parameter(value);
		components.put(name, variable);
		variable.setDescription(name);
		return this;
	}
	
	public ModelBuilder constant(String name, Tensor value) {
		if(components.containsKey(name)) {
			((Constant)components.get(name)).setTo(value);
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
	
	public ModelBuilder operation(String desc) {
		String[] lines = desc.split("\\;|\\\n");
		if(lines.length>1) {
			for(String line : lines)
				if(!line.trim().isEmpty())
					operation(line);
			return this;
		}
		desc = desc.trim();
		desc = desc.replace("=", " = ");
		desc = desc.replace(".", " . ");
		desc = desc.replace("+", " + ");
		desc = desc.replace("*", " * ");
		desc = desc.replace("[", " [ ");
		desc = desc.replace("]", " ] ");
		desc = desc.replace("(", " ( ");
		desc = desc.replace(")", " ) ");
		desc = desc.replace(",", " , ");
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
							if(components.containsKey(arg)) 
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
						newDesc += "( "+args+" )";
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
			String[] operators = {" + ", " . ", " * "};
			for(String operator : operators) {
				if(madeChanges)
					break;
				String[] splt = desc.split("\\s*=\\s*");
				if(splt.length!=2)
					throw new RuntimeException("Exactly one equality needed in each operation: "+desc);
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
							if(components.containsKey(arg)) 
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
					if(components.containsKey(arg)) 
						newDesc += arg;
					else {
						newDesc += "("+arg+")";
						madeChanges = true;
					}
				}
				desc = splt[0]+" = "+newDesc;
			}
		}
		//System.out.println(desc);
		
		routing += desc + "\n";
		desc = desc.replace("(", "").replace(")", "").replace(",", "")+" ";
		
		
		String[] splt = desc.split("\\s+");
		String name = splt[0];
		String arg0 = null;
		String arg1 = null;
		assertValidName(name);
		
		NNOperation component;
		if(splt[3].equals("+")) {
			component = new Add();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[3].equals("x")) {
			component = new Repeat();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[2].equals("sum")) {
			component = new Sum();
			arg0 = splt[3];
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
		else if(splt[2].equals("sigmoid")) {
			component = new Sigmoid();
			arg0 = splt[3];
		}
		else if(splt[2].equals("repeat")) {
			component = new Repeat();
			arg0 = splt[3];
			arg1 = splt[4];
		}
		else if(splt[3].equals(".")) {
			component = new Multiply();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[3].equals("*")) {
			component = new MatMul();
			arg0 = splt[2];
			arg1 = splt[4];
		}
		else if(splt[3].equals("[")) {
			component = new Gather();
			arg0 = splt[4];
			arg1 = splt[2];
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
				System.err.println("The component "+component.describe()+" was not added by this builder");
		for(NNOperation component : actualComponents)
			if(!allFoundComponents.contains(component)) {
				System.err.println("The component "+component.describe()+" does not lead to an output and will be removed from the outputs of other components");
				for(NNOperation other : actualComponents)
					other.getOutputs().remove(component);
			}
		
		return this;
	}
	
	public String describe() {
		return routing;
	}
	public ModelBuilder print() {
		System.out.println(describe());
		return this;
	}
	public ModelBuilder printState() {
		for(NNOperation component : components.values())
			System.out.println(component.view());
		return this;
	}
}
