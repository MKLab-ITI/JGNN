package mklab.JGNN.core;

/**
 * This interface abstracts a probability distribution passed that can be passed to {@link Tensor#setToRandom(Distribution)}
 * for random tensor initialization.
 * 
 * @author Emmanouil Krasanakis
 */
public interface Distribution {
	public double sample();
}
