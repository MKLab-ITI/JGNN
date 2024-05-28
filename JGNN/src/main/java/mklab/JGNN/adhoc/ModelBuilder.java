package mklab.JGNN.adhoc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.tensor.SparseTensor;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.nn.activations.Exp;
import mklab.JGNN.nn.activations.L1;
import mklab.JGNN.nn.activations.LRelu;
import mklab.JGNN.nn.activations.NExp;
import mklab.JGNN.nn.activations.PRelu;
import mklab.JGNN.nn.activations.Relu;
import mklab.JGNN.nn.activations.Sigmoid;
import mklab.JGNN.nn.activations.Tanh;
import mklab.JGNN.nn.inputs.Constant;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.inputs.Variable;
import mklab.JGNN.nn.operations.Add;
import mklab.JGNN.nn.operations.Attention;
import mklab.JGNN.nn.operations.Complement;
import mklab.JGNN.nn.operations.Concat;
import mklab.JGNN.nn.operations.Dropout;
import mklab.JGNN.nn.operations.From;
import mklab.JGNN.nn.operations.Gather;
import mklab.JGNN.nn.operations.Identity;
import mklab.JGNN.nn.operations.Log;
import mklab.JGNN.nn.operations.MatMul;
import mklab.JGNN.nn.operations.Multiply;
import mklab.JGNN.nn.operations.Reduce;
import mklab.JGNN.nn.operations.Repeat;
import mklab.JGNN.nn.operations.Reshape;
import mklab.JGNN.nn.operations.To;
import mklab.JGNN.nn.operations.Transpose;
import mklab.JGNN.nn.pooling.SoftMax;
import mklab.JGNN.nn.pooling.Sort;
import mklab.JGNN.nn.pooling.Sum;
import mklab.JGNN.nn.pooling.Max;
import mklab.JGNN.nn.pooling.Mean;

/**
 * This class and subclasses can be used to create {@link Model} instances 
 * by automatically creating and managing {@link NNOperation} instances based on
 * textual descriptions.
 * 
 * @author Emmanouil Krasanakis
 * @see #config(String, double)
 * @see #constant(String, double)
 * @see #constant(String, Tensor)
 * @see #operation(String)
 * @see #out(String)
 * @see #getModel()
 */
public class ModelBuilder {
	private String routing = "";
	private Model model = null;
	private HashMap<String, NNOperation> components = new HashMap<String, NNOperation>();
	private HashMap<String, Double> configurations = new HashMap<String, Double>();
	private HashMap<String, String> functions = new HashMap<String, String>();
	private HashMap<String, String> functionSignatures = new HashMap<String, String>();
	private HashMap<String, Integer> functionUsages = new HashMap<String, Integer>();
	private int tmpVariableIdentifier = 0;
	public ModelBuilder() {
		this(new Model());
		configurations.put("?", 0.0);
	}
	public ModelBuilder(Model model) {
		this.model = model;
	}
	/**
	 * Retrieves the model currently built by the builder.
	 * This can changed depending on additional building method calls.
	 * @return A {@link Model} instance.
	 */
	public Model getModel() {
		return model;
	}
	public ModelBuilder save(Path path) {
		try(BufferedWriter writer = Files.newBufferedWriter(path)){
			writer.write(this.getClass().getCanonicalName()+"\n");
			for(String configurationName : configurations.keySet())
				writer.write(configurationName+" = config: "+configurations.get(configurationName)+"\n");
			for(String componentName : components.keySet())
				if(components.get(componentName) instanceof Parameter) {
					if(components.get(componentName) instanceof Variable) {
						writer.write(componentName+" = var: null\n");
						continue;
					}
					writer.write(componentName+" = ");
					Tensor value = ((Parameter)components.get(componentName)).get();
					writer.write((((Parameter)components.get(componentName)).isConstant()?"const ":"param ")+value.describe()+": ");
					if(value.density()<0.3) {
						writer.write("{");
						boolean isNotFirst = false;
						for(long pos : value.getNonZeroElements()) {
							if(isNotFirst)
								writer.write(",");
							writer.write(pos+":"+value.get(pos));
							isNotFirst = true;
						}
						writer.write("}\n");
					}
					else {
						writer.write("[");
						for(long pos=0;pos<value.size();pos++) {
							if(pos==0)
								writer.write(""+value.get(pos));
							else
								writer.write(","+value.get(pos));
						}
						writer.write("]\n");
					}
				}
			writer.write(routing+"\n");
			writer.write(saveCommands());
		}
		catch(IOException ex){
			System.err.println(ex.toString());
			return null;
		}
		return this;
	}
	
	public static ModelBuilder load(Path path) {
		ModelBuilder builder;
		try(BufferedReader reader = Files.newBufferedReader(path)){
			String line = reader.readLine();
			try {
			    builder = (ModelBuilder) Class.forName(line).getDeclaredConstructor().newInstance();
			} 
			catch (Exception e) {
				e.printStackTrace();
				return null;
			}
			while((line = reader.readLine())!=null) {
				if(line.length()==0)
					continue;
				int eqPosition = line.indexOf('=');
				if(eqPosition==-1) {
					String[] splt = line.split("\\s+", 2);
					if(splt.length!=2 || !builder.loadCommand(splt[0], splt[1]))
						throw new IOException("Unidentified command: "+line+". A different JGNN version was likely used to save the model.");
					continue;
				}
				int initPosition = line.indexOf(':', eqPosition);
				String name = line.substring(0, eqPosition-1);
				if(builder.components.containsKey(name))
					continue;
				if(initPosition==-1) {
					System.out.println("parsing " + line);
					builder.operation(line);
					continue;
				}
				String type = line.substring(eqPosition+2, initPosition);
				System.out.println("reading " + name+" "+type);
				if(type.equals("var"))
					builder.var(name);
				else if(type.equals("out"))
					builder.out(name);
				else if(type.equals("config")) 
					builder.config(name, Double.parseDouble(line.substring(initPosition+1)));
				else if(type.contains("Tensor ") || type.contains("Matrix ")) {
					boolean isDense = line.charAt(initPosition+2)=='[';
					Tensor tensor;
					if(type.contains("Tensor ")) {
						String[] dimParts = type.substring(type.indexOf('(')+1, type.lastIndexOf(')')).split("\\s", 2);
						int dim = Integer.parseInt(dimParts[dimParts.length-1]);
						tensor = isDense?new DenseTensor(dim):new SparseTensor(dim);
						if(dimParts.length>1)
							tensor.setDimensionName(dimParts[0]);
					}
					else {
						String[] dims = type.substring(type.indexOf('(')+1, type.lastIndexOf(')')).split(",");
						String[] dimRowParts = dims[0].trim().split("\\s", 2);
						int dimRow = Integer.parseInt(dimRowParts[dimRowParts.length-1]);
						String[] dimColParts = dims[1].trim().split("\\s", 2);
						int dimCol = Integer.parseInt(dimColParts[dimColParts.length-1]);
						tensor = isDense?new DenseMatrix(dimRow, dimCol):new SparseMatrix(dimRow, dimCol);
						if(dimRowParts.length>1)
							tensor.cast(Matrix.class).setRowName(dimRowParts[0]);
						if(dimColParts.length>1)
							tensor.cast(Matrix.class).setColName(dimColParts[0]);
					}
					if(line.charAt(initPosition+2)=='[') {
						long idx = 0;
						String accum = "";
						for(int pos=initPosition+3;pos<line.length();pos++) {
							char c = line.charAt(pos);
							if(c==']' || c==',') {
								tensor.put(idx, Double.parseDouble(accum));
								idx += 1;
								accum = "";
							}
							else
								accum += c;
							if(c==']')
								break;
						}
					}
					else if(line.charAt(initPosition+2)=='{') {
						String accum = null;
						String key = "";
						for(int pos=initPosition+3;pos<line.length();pos++) {
							char c = line.charAt(pos);
							if(c==']' || c==',') {
								tensor.put(Long.parseLong(key), Double.parseDouble(accum));
								accum = null;
								key = "";
							}
							else if(c==':')
								accum = "";
							else if(accum!=null)
								accum += c;
							else
								key += c;
						}
					}
					if(type.startsWith("const "))
						builder.constant(name, tensor);
					else
						builder.param(name, tensor);
				}
				else
					throw new IOException("Unidentified primitive: "+type+". A different JGNN version was likely used to save the model.");
			}
		}
		catch(IOException ex){
			System.err.println(ex.toString());
			return null;
		}
		return builder;
	}
	
	protected String saveCommands() {
		String ret = "";
		for(String componentName : components.keySet())
			if(model.getOutputs().contains(components.get(componentName)))
				ret += "return "+componentName+"\n";
		return ret;
	}
	
	protected boolean loadCommand(String command, String data) {
		if(command.equals("return")) {
			out(data);
			return true;
		}
		return false;
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
	
	/**
	 * Declares a component with the given name to be used as an input 
	 * of the managed model.
	 * @param name The name of the component.
	 * @return The builder's instance.
	 */
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
		if(name.contains("(") || name.contains("[")) {
			operation("_return = "+name);
			name = "_return";
		}
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
		this.configurations.put(name, value);
		return this;
	}
	
	public int getConfigOrDefault(String name, int defaultValue) {
		return (int)(double)configurations.getOrDefault(name, (double) defaultValue);
	}
	
	public double getConfigOrDefault(String name, double defaultValue) {
		return configurations.getOrDefault(name, defaultValue);
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
	 * Declares a learnable {@link mklab.JGNN.nn.inputs.Paramater} component with the given name,
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
	
	/**
	 * Declares a non-learnable constant component with the given name.
	 * This can be used in computations. To edit the constant's values,
	 * use {@link #get(String)} to retrieve the respective component.
	 * @param name The name of the constant component.
	 * @param value A double value to assign to the constant.
	 * @return The builder's instance.
	 * @see #config(String, double)
	 * @see #constant(String, Tensor)
	 */
	public ModelBuilder constant(String name, double value) {
		return constant(name, Tensor.fromDouble(value));
	}
	
	/**
	 * Declares a non-learnable constant component with the given name.
	 * This can be used in computations. To edit the constant's values,
	 * use {@link #get(String)} to retrieve the respective component.
	 * @param name The name of the constant component.
	 * @param value A Tensor value to assign to the constant.
	 * @return The builder's instance.
	 * @see #constant(String, double)
	 */
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
	
	/**
	 * Retrieves the {@link NNOperation} registered with the provided
	 * name, for example to investigates its value.
	 * @param name The name of the component.
	 * @return A {@link NNOperation}.
	 */
	public NNOperation get(String name) {
		return components.get(name);
	}
	
	/**
	 * This is a wrapper for <code>getModel().predict(inputs)</code>
	 * <b>without</b> returning output values (use {@link #get(String)}
	 * afterwards to view outputs.
	 * @param inputs A variable number of Tensor inputs.
	 * @return The builder's instance.
	 * @see #getModel()
	 * @see Model#predict(List)
	 */
	public ModelBuilder runModel(Tensor... inputs) {
		model.predict(inputs);
		return this;
	}
	/**
	 * This is a wrapper for <code>getModel().predict(inputs)</code>
	 * <b>without</b> returning output values (use {@link #get(String)}
	 * afterwards to view outputs.
	 * @param inputs A list of Tensor inputs.
	 * @return The builder's instance.
	 * @see #getModel()
	 * @see Model#predict(ArrayList<Tensor>)
	 */
	public ModelBuilder runModel(ArrayList<Tensor> inputs) {
		model.predict(inputs);
		return this;
	}
	
	public ModelBuilder function(String name, String value) {
		value = value.trim();
		if(value.indexOf(")")==-1)
			throw new RuntimeException("Function signature should be enclosed in parentheses.");
		if(value.indexOf("{")==-1)
			throw new RuntimeException("Function body should start with brackets.");
		functions.put(name, value.substring(value.indexOf("{")+1, value.length()-1));
		functionSignatures.put(name, value.substring(1, value.indexOf(")")));
		functionUsages.put(name, 0);
		return this;
	}
	
	private static List<String> extractTokens(String input) {
        String tokenRegex = "\\b[a-zA-Z_][a-zA-Z0-9_]*\\b"+"|\\b\\w+\\b|\\(|\\)|\\=|\\+|\\;|\\!|\\:|\\#|\\-|\\.|\\*|\\@|\\/|\\[|\\]|\\,|\\?|\\||\\{|\\}";
        Pattern tokenPattern = Pattern.compile(tokenRegex);
        Matcher tokenMatcher = tokenPattern.matcher(input);
        List<String> tokens = new ArrayList<>();
        while (tokenMatcher.find()) {
            String token = tokenMatcher.group();
            tokens.add(token);
        }
        return tokens;
    }
	

    private static boolean isNumeric(String str) {
        if (str == null || str.isEmpty()) {
            return false;
        }
        String regex = "[+-]?\\d*(\\.\\d+)?";
        return str.matches(regex);
    }
	
	/**
	 * Parses one or more operations split by new line characters or ; 
	 * to add to the execution graph. All operations should assign a
	 * value to a new component name and comprise operators and functions. 
	 * For a detailed description of the domain-specific language this
	 * method accepts, please refer to the library's 
	 * <a href="https://github.com/MKLab-ITI/JGNN/blob/main/tutorials/NN.md">
	 * online documentation</a>.
	 * @param desc The operation to parse.
	 * @return The builder's instance.
	 * @see #config(String, double)
	 * @see #constant(String, double)
	 * @see #constant(String, Tensor)
	 * @see #out(String)
	 */
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
									|| (newDesc.endsWith(" sort ") && isDouble(arg)) 
									|| (newDesc.endsWith(" reshape ") && isDouble(arg))
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
			String[] operators = {" + ", " * ", " @ ", " | ", "-", "/"};
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
				return this;
			}
			catch(NumberFormatException e) {
				component = new Identity();
				arg0 = splt[2];
				//throw new RuntimeException("Symbol "+splt[2]+" not defined.");
			}
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
		else if(splt[2].equals("mean")) {
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
			component = new Mean(mode);
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
		else if(splt[2].equals("sort")) {
			component = new Sort((int)(splt.length>4?parseConfigValue(splt[4]):0))
					.setDimensionName(splt.length<=4 || isDouble(splt[4])?null:splt[4]);
			arg0 = splt[3];
		}
		else if(splt[2].equals("reshape")) {
			component = new Reshape((long)(splt.length>4?parseConfigValue(splt[4]):1),
					(long)(splt.length>5?parseConfigValue(splt[5]):1))
					.setDimensionName(splt.length>4&&isDouble(splt[4])?null:splt[4],
							splt.length<=5 || isDouble(splt[5])?null:splt[5]
							);
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
		else if(splt[2].equals("monitor")) {
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
		else if(splt[2].equals("L1")) {
			boolean mode = false;
			if(splt.length>4) {
				String modeText = splt[4].trim();
				if(modeText.equals("col"))
					mode = false;
				else if(modeText.equals("row"))
					mode = true;
				else
					throw new RuntimeException("Invalid argument "+modeText+" to L1");
			}
			component = new L1(mode);
			arg0 = splt[3];
		}
		else if(splt[2].equals("exp")) {
			component = new Exp();
			arg0 = splt[3];
		}
		else if(splt[2].equals("nexp")) {
			component = new NExp();
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
		else if(splt[2].equals("attention") || splt[2].equals("att")) {
			component = new Attention();
			arg0 = splt[3];
			arg1 = splt[4];
		}
		else if(splt[2].equals("transpose")) {
			component = new Transpose();
			arg0 = splt[3];
		}
		else if(splt[2].equals("from")) {
			component = new From();
			arg0 = splt[3];
		}
		else if(splt[2].equals("to")) {
			component = new To();
			arg0 = splt[3];
		}
		else if(splt[2].equals("reduce")) {
			component = new Reduce();
			arg0 = splt[3];
			arg1 = splt[4];
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
		else if(functions.containsKey(splt[2])) {
			String[] args = functionSignatures.get(splt[2]).split("\\,");
			//if(args.length!=splt.length-3)
			//	throw new RuntimeException("Function "+splt[2]+" requires at most "+args.length+" arguments");
			int functionRepetition = functionUsages.get(splt[2]);
			functionUsages.put(splt[2], functionRepetition+1);
			HashMap<String, Double> configStack = this.configurations;
			this.configurations = new HashMap<String, Double>(this.configurations);
			HashMap<String, String> customNames = new HashMap<String, String>();
			for(int i=0;i<args.length;i++)
				if(!args[i].contains(":"))
					customNames.put(args[i].trim(), splt[i+3]);
				else {
					String config = args[i].substring(0, args[i].indexOf(":")).trim();
					String value = args[i].substring(args[i].indexOf(":")+1).trim();
					if(value.equals("extern")) {
						if(!this.configurations.containsKey(config))
							throw new RuntimeException("Required external config: "+config);
					}
					if(!this.configurations.containsKey(config))
						this.config(config, parseConfigValue(value));
					/*// these are parsed in the attempt to create an intermediate variable for the argument
					  if(i<splt.length-3) {
						String config = splt[i+3].substring(0, splt[i+3].indexOf(":")).trim();
						String value = splt[i+3].substring(splt[i+3].indexOf(":")+1).trim();
						if(value.equals("extern")) {
							if(!this.configurations.containsKey(config))
								throw new RuntimeException("Required external config: "+config);
						}
						else
							this.config(config, parseConfigValue(value));
					}*/
				}
			List<String> tokens = extractTokens(functions.get(splt[2]));
			HashSet<String> keywords = new HashSet<String>();
			keywords.addAll(functions.keySet());
			keywords.addAll(Arrays.asList(".", "+", "-", "*", "/", "@", ",", "(", ")", ";", "=",
					"max", "min", "vector", "matrix", "vec", "mat", "[", "]", "{", "}", "|", "#", "!", ":", 
					"extern", "softmax",
					"from", "to", "reduce", "transpose", "attention", "att", "dropout", "drop", 
					"repeat", "exp", "nexp", "L1", "sigmoid", "transpose", "monitor", 
					"log", "tanh", "prelu", "lrelu", "relu", "reshape", "mean", "col", "row"));
			keywords.addAll(this.components.keySet());
			keywords.addAll(this.configurations.keySet());
			customNames.put("return", splt[0]+" = ");
			String newExpr = "";
			routing = prevRouting; // remove the function call from the routing
			boolean prevHash = false;
			boolean prevTemp = false;
			HashMap<String, Integer> temp = new HashMap<String, Integer>();
			HashMap<String, String> renameLater = new HashMap<String, String>();
			for(int i=0;i<tokens.size();i++) {
				String token = tokens.get(i);
				if(i<tokens.size()-1 && tokens.get(i+1).equals("=")) {
					int id = temp.getOrDefault(token, 0);
					temp.put(token, id+1);
					renameLater.put(token, "_"+splt[2]+functionRepetition+"_stack"+id+"_"+token);
					token = "_"+splt[2]+functionRepetition+"_stack"+id+"_"+token;
				}
				else if(customNames.containsKey(token))
					token = customNames.get(token);
				else if(!keywords.contains(token) && !isNumeric(token) && !prevHash) 
					token = "_"+splt[2]+functionRepetition+"_"+token;
				prevHash = token.equals("#");
				prevTemp = token.equals("!");
				if(token.equals(";") || token.equals("}")) {
					customNames.putAll(renameLater);
					renameLater.clear();
				}
				if(!prevHash && !prevTemp)
					newExpr += token;
			}
			this.operation(newExpr);
			this.configurations = configStack;
			return this;
		}
		else
			throw new RuntimeException("Invalid operation: "+desc);
		
		if(arg0.contains(":")) {
			String config = arg0.substring(0, arg0.indexOf(":")).trim();
			String value = arg0.substring(arg0.indexOf(":")+1).trim();
			if(value.equals("extern")) {
				if(!this.configurations.containsKey(config))
					throw new RuntimeException("Required external config: "+config);
			}
			this.config(config, parseConfigValue(value));
			return this;
		}

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
	public ModelBuilder autosize(Tensor... inputs) {
		createForwardValidity(Arrays.asList(inputs));
		assertBackwardValidity();
		return this;
	}

	public ModelBuilder autosize(List<Tensor> inputs) {
		createForwardValidity(inputs);
		assertBackwardValidity();
		return this;
	}

	/**
	 * Asserts that all components parsed into a call graph with
	 * {@link #operation(String)} are eventually used by at least one {@link #out(String)}
	 * component.
	 * @return The builder's instance.
	 * @throws RuntimeException if not all execution graph branches lead to declared outputs.
	 */
	public ModelBuilder createForwardValidity(List<Tensor> inputs) {
		if(inputs.size() != model.getInputs().size())
			throw new IllegalArgumentException("Incompatible input size: expected"+model.getInputs().size()+" inputs instead of "+inputs.size());
		for(NNOperation output : model.getOutputs())
			output.clearPrediction();
		for(int i=0;i<inputs.size();i++)
			model.getInputs().get(i).setTo(inputs.get(i));
		for(int i=0;i<model.getOutputs().size();i++)
			model.getOutputs().get(i).runPredictionAndAutosize();
		return this;
	}
	
	
	/**
	 * Asserts that all components parsed into a call graph with
	 * {@link #operation(String)} are eventually used by at least one {@link #out(String)}
	 * component.
	 * @return The builder's instance.
	 * @throws RuntimeException if not all execution graph branches lead to declared outputs.
	 */
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
	/**
	 * Creates a description of the builded model's internal execution graph.
	 * @return A <code>String</code>.
	 * @see #print()
	 */
	public String describe() {
		getModel();
		return routing;
	}
	/**
	 * Exports the builded model's execution graph into a <i>.dot</i> format
	 * representation.
	 * @return A <code>String</code> to be pasted into GraphViz for visualization.
	 */
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
					+"[label=\""+component.getDescription()+" = "+component.getSimpleDescription()+"\", shape=doubleoctagon]";
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
				ret+="\n   "+component.getDescription()+"[label=\""+component.getSimpleDescription()+"\"]";
			}
			else
				ret+="\n   "+component.getDescription()+"[label=\""+component.getDescription()+" = "+component.getSimpleDescription()+"\"]";
			
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
