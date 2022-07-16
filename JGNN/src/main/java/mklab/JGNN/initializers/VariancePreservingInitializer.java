package mklab.JGNN.initializers;

import mklab.JGNN.core.Distribution;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelInitializer;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.inputs.Variable;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class describes a broad class of {@link ModelInitializer} strategies, in which
 * dense neural layer initialization is controlled so that variance is mostly preserved from
 * inputs to outputs to avoid vanishing or exploding gradients in the first training
 * runs. 
 * <br>
 * This initializer traverses the execution tree to discover the impact of matrix parameters
 * to output variances, as eventually determined by backtracking
 * {@link NNOperation#getNonLinearity(int, double, double)} up to non-linear components,
 * where the latter are identified by the condition <code>getNonLinearity(0, 1, 1)!=1</code>.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class VariancePreservingInitializer extends ModelInitializer {
	protected abstract Distribution getDistribution(long rows, long cols, double gain);
	double defaultGain = 0;
	@Override
	public Model apply(Model model) {
		HashMap<NNOperation, Integer> countEventualParameters = new HashMap<NNOperation, Integer>();
		ArrayList<NNOperation> depthFirst = model.getDepthLastOperations();
		java.util.Collections.reverse(depthFirst);
		for(NNOperation operation : depthFirst) {
			if(operation instanceof Parameter 
					&& !(operation instanceof Variable)
					&& !operation.isConstant()
					&& ((Parameter)operation).get() instanceof Matrix)
				countEventualParameters.put(operation, 1);
			if(operation.getNonLinearity(0, 1, 1)!=1)
				countEventualParameters.put(operation, 0);
			else
				for(NNOperation input : operation.getInputs())
					countEventualParameters.put(operation, 
							countEventualParameters.getOrDefault(operation, 0)
							+countEventualParameters.getOrDefault(input, 0));
		}
		
		
		HashMap<NNOperation, Double> gains = new HashMap<NNOperation, Double>();
		for(NNOperation operation : model.getDepthLastOperations()) {
			if(operation instanceof Parameter 
					&& !operation.isConstant()
					&& !(operation instanceof Variable)) {
				Parameter parameter = (Parameter)operation;
				if(parameter.get() instanceof Matrix) {
					Matrix mat = parameter.get().cast(Matrix.class);
					mat.setToRandom(getDistribution(mat.getRows(), mat.getCols(), 
							gains.get(operation)));
				}
				else
					parameter.get().setToZero();
			}
			for(int i=0;i<operation.getInputs().size();i++) {
				if(countEventualParameters.get(operation)==0)
					continue;
				double inputMass = 
						countEventualParameters.getOrDefault(operation.getInputs().get(i), 0) 
						/ (double)countEventualParameters.get(operation);
				NNOperation input = operation.getInputs().get(i);
				gains.put(input, gains.getOrDefault(input, 0.)+operation.getNonLinearity(i, inputMass, gains.getOrDefault(operation, 1.)));
				//System.out.println(input.describe());
			}
			
		}
			
		/*
		for(Parameter parameter : model.getParameters())
			if(parameter.get() instanceof Matrix) {
				Matrix mat = parameter.get().cast(Matrix.class);
				mat.setToRandom(getDistribution(mat.getRows(), mat.getCols(), Math.sqrt(2)));
			}
			else
				parameter.get().setToZero();*/
		return model;
	}
}
