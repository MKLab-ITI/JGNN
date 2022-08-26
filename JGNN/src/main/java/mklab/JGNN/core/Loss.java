package mklab.JGNN.core;

/**
 * This class provides an abstract implementation of loss functions
 * to be used during {@link Model} training. Preferred use is by 
 * passing loss instances to {@link ModelTraining}s.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class Loss {
	public abstract double evaluate(Tensor output, Tensor desired);
	public abstract Tensor derivative(Tensor output, Tensor desired);
}
