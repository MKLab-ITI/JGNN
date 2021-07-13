package mklab.JGNN.nn.optimizers;

import java.util.HashMap;

import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Implements a gradient descent {@link Optimizer}. It supports degrading learning rates.
 * 
 * @author Emmanouil Krasanakis
 */
public class GradientDescent implements Optimizer {
	protected double learningRate;
	private double degradation;
	private HashMap<Tensor, Double> individualLearningRates = new HashMap<Tensor, Double>();
	
	/**
	 * Initializes a {@link GradientDescent} optimizer with fixed learning rate.
	 * @param learningRate The learning rate.
	 */
	public GradientDescent(double learningRate) {
		this(learningRate, 1);
	}

	/**
	 * Initializes a {@link GradientDescent} optimizer with degrading learning rate.
	 * @param learningRate The learning rate.
	 * @param degradation The quantity to multiply each tensor's learning rate with after each iteration.
	 */
	public GradientDescent(double learningRate, double degradation) {
		this.learningRate = learningRate;
		this.degradation = degradation;
	}
	protected GradientDescent() {}
	@Override
	public void update(Tensor value, Tensor gradient) {
		synchronized(value) {
			individualLearningRates.put(value, individualLearningRates.getOrDefault(value, learningRate)*degradation);
			value.selfAdd(gradient.multiply(-individualLearningRates.get(value)));
		}
	}
	@Override
	public void reset() {
		individualLearningRates = new HashMap<Tensor, Double>();
	}
}
