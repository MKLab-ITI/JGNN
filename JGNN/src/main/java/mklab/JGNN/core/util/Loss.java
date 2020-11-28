package mklab.JGNN.core.util;

import mklab.JGNN.core.Tensor;

/**
 * Provides computation and (partial) derivation of popular activation functions
 * and cross-entropy loss functions.
 * 
 * @author Emmanouil Krasanakis
 */
public interface Loss {
	/**
	 * The sigmoid function 1/(1+exp(-x)).
	 * @param x The activation of the sigmoid function.
	 * @return The sigmoid value.
	 * @see #sigmoid(Tensor)
	 */
	public static double sigmoid(double x) {
		return 1./(1+Math.exp(-x));
	}
	
	/**
	 * The tanh activation (exp(x)-exp(-x))/(exp(x)+exp(-x))
	 * @param x  The activation of the tanh function.
	 * @return The tanh value.
	 * @see #tanh(Tensor)
	 */
	public static double tanh(double x) {
		return (1-Math.exp(-2*x))/(1+Math.exp(-2*x));
	}
	
	/**
	 * The relu activation x if x > 0, 0 otherwise
	 * @param x  The activation of the relu function.
	 * @return The relu value.
	 * @see #relu(Tensor)
	 */
	public static double relu(double x) {
		return x>0?x:0;
	}
	
	/**
	 * The derivative of the {@link #sigmoid(double)} function.
	 * @param x The activation of the sigmoid function.
	 * @return The sigmoid derivative's value.
	 * @see #sigmoidDerivative(Tensor)
	 */
	public static double sigmoidDerivative(double x) {
		double sigma = sigmoid(x);
		return sigma*(1-sigma);
	}

	/**
	 * The derivative of the {@link #tanh(double)} function.
	 * @param x The activation of the tanh function.
	 * @return The tanh derivative's value.
	 * @see #sigmoidDerivative(Tensor)
	 */
	public static double tanhDerivative(double x) {
		double tanhValue = tanh(x);
		return 1-tanhValue*tanhValue;
	}

	/**
	 * The derivative of the {@link #relu(double)} function.
	 * @param x The activation of the relu function.
	 * @return The relu derivative's value.
	 * @see #reluDerivative(Tensor)
	 */
	public static double reluDerivative(double x) {
		return x>=0?1:0;
	}
	
	/**
	 * A cross entropy loss for one sample computes as -label*log(output) -(1-label)*log(1-output). To avoid producing invalid
	 * values, an eps of 1.E-12 is used to constraint the cross entropy in the range [-12, 12].
	 * @param output The output of a prediction task. Should lie in the range [0,1]
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy value.
	 * @throws IllegalArgumentException
	 */
	public static double crossEntropy(double output, double label) {
		if(label!=0 && label!=1)
			throw new IllegalArgumentException("Only binary labels are allowed for computing the cross entropy loss");
		if(output<0 || output>1)
			throw new IllegalArgumentException("The predicted output passed on to cross entropy should lie in the range [0,1]");
		return -label*Math.log(output+1.E-12) - (1-label)*Math.log(1-output+1.E-12);
	}

	/**
	 * The derivative of the {@link #crossEntropy(double, double)} loss. To avoid producing invalid
	 * values, an eps of 1.E-12 is used to constraint the cross entropy in the range [-12, 12], which results
	 * to this derivative being constrained in the range [-1.E12, 1.E12].
	 * @param output The output of a prediction task. Should lie in the range [0,1]
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy derivative's value.
	 * @throws IllegalArgumentException
	 */
	public static double crossEntropyDerivative(double output, double label) {
		if(label!=0 && label!=1)
			throw new IllegalArgumentException("Only binary labels are allowed for computing the cross entropy loss");
		if(output<0 || output>1)
			throw new IllegalArgumentException("The predicted output passed on to cross entropy should lie in the range [0,1]");
		return -label/(output+1.E-12) + (1-label)/(1-output+1.E-12);
	}
	
	/**
	 * The derivative of <code>crossEntropy(sigmoid(x), label)</code> with respect to x. This function can avoid
	 * using an eps and is hence more precise than the expression
	 * <code>crossEntropyDerivative(sigmoid(x), label)*sigmoidDerivative(x)</code>.
	 * @param x The activation of the sigmoid function.
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy partial derivative with respect to the activation passed to an intermediate sigmoid transformation.
	 * @throws IllegalArgumentException
	 */
	public static double crossEntropySigmoidDerivative(double x, double label) {
		if(label!=0 && label!=1)
			throw new IllegalArgumentException("Only binary labels are allowed for computing the cross entropy loss");
		double sigma = sigmoid(x);
		return -label*(1-sigma) + (1-label)*sigma;
	}
	

	/**
	 * The derivative of <code>crossEntropy(tanh(x), label)</code> with respect to x. This function calculates
	 * <code>crossEntropyDerivative(tanh(x), label)*tanhDerivative(x)</code>.
	 * @param x The activation of the tanh function.
	 * @param label The desired label of the prediction task. Should assume binary values 0 or 1
	 * @return The cross entropy partial derivative with respect to the activation passed to an intermediate tanh transformation.
	 */
	public static double crossEntropyTanhDerivative(double x, double label) {
		double tanhValue = tanh(x);
		return crossEntropyDerivative(tanhValue, label)*(1-tanhValue*tanhValue);
	}
	

	/**
	 * Applies {@link #sigmoid(double)} element-by-element.
	 * @param x  The activation tensor of the sigmoid function.
	 * @return The tensor of sigmoid values.
	 */
	public static Tensor sigmoid(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements())
			ret.put(i, sigmoid(x.get(i)));
		return ret;
	}

	/**
	 * Applies {@link #tanh(double)} element-by-element.
	 * @param x  The activation tensor of the tanh function.
	 * @return The tensor of tanh values.
	 */
	public static Tensor tanh(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements())
			ret.put(i, tanh(x.get(i)));
		return ret;
	}

	/**
	 * Applies {@link #relu(double)} element-by-element.
	 * @param x  The activation tensor of the relu function.
	 * @return The tensor of relu values.
	 */
	public static Tensor relu(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements())
			ret.put(i, relu(x.get(i)));
		return ret;
	}
	
	/**
	 * Applies {@link #sigmoidDerivative(double)} function.
	 * @param x The activation tensor of the sigmoid function.
	 * @return The tensor of sigmoid derivative values.
	 */
	public static Tensor sigmoidDerivative(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements())
			ret.put(i, sigmoidDerivative(x.get(i)));
		return ret;
	}

	/**
	 * Applies {@link #tanhDerivative(double)} function.
	 * @param x The activation tensor of the tanh function.
	 * @return The tensor of tanh derivative values.
	 */
	public static Tensor tanhDerivative(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements())
			ret.put(i, tanhDerivative(x.get(i)));
		return ret;
	}

	/**
	 * Applies {@link #reluDerivative(double)} function.
	 * @param x The activation tensor of the relu function.
	 * @return The tensor of relu derivative values.
	 */
	public static Tensor reluDerivative(Tensor x) {
		Tensor ret = x.zeroCopy();
		for(long i : x.getNonZeroElements())
			ret.put(i, reluDerivative(x.get(i)));
		return ret;
	}
}
