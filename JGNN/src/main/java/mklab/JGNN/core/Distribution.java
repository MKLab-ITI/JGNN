package mklab.JGNN.core;

/**
 * This interface abstracts a probability distribution that can be passed to
 * {@link Tensor#setToRandom(Distribution)} for random tensor initialization.
 * 
 * @author Emmanouil Krasanakis
 */
public interface Distribution {
	/**
	 * Sets the distribution's seed. This should yield reproducible sampling.
	 * 
	 * @param seed The distribution's new seed.
	 * @return <code>this</code> Distribution.
	 */
	public Distribution setSeed(long seed);

	/**
	 * Retrieves a new sample from the distribution.
	 * 
	 * @return A double value.
	 */
	public double sample();

	/**
	 * Sets the mean of the distribution.
	 * 
	 * @param mean The new mean.
	 * @return <code>this</code> Distribution.
	 */
	public Distribution setMean(double mean);

	/**
	 * Sets the standard deviation of the distribution.
	 * 
	 * @param std The new standard deviation.
	 * @return <code>this</code> Distribution.
	 */
	public Distribution setDeviation(double std);

	/**
	 * Retrieves the distribution's mean.
	 * 
	 * @return The mean value.
	 */
	public double getMean();

	/**
	 * Retrieves the distribution's standard deviation.
	 * 
	 * @return The standard deviation.
	 */
	public double getDeviation();
}
