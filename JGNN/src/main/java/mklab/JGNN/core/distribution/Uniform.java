package mklab.JGNN.core.distribution;

import java.util.Random;

import mklab.JGNN.core.Distribution;

/**
 * Implements a Uniform {@link Distribution} of given bounds.
 * @author Emmanouil Krasanakis
 */
public class Uniform implements Distribution {
	private double from;
	private double to;
	private Random randomGenerator;
	private static double sqrt12 = Math.sqrt(12);
	
	/**
	 * Instantiates a uniform distribution that samples values from the range [0,1].
	 */
	public Uniform() {
		this(0, 1);
	}
	
	/**
	 * Instantiates a uniform distribution that samples values from the given range [from, to].
	 * @param from The minimum value of the distribution.
	 * @param to The maximum value of the distribution.
	 */
	public Uniform(double from, double to) {
		setRange(from, to);
		randomGenerator = null;
	}
	
	/**
	 * Sets the random of the uniform distribution.
	 * @param from The range's start.
	 * @param to The range's end.
	 * @return <code>this</code> Distribution.
	 */
	public Uniform setRange(double from, double to) {
		if(from>to)
			throw new IllegalArgumentException("Invalid distribution range");
		this.from = from;
		this.to = to;
		return this;
	}

	@Override
	public Uniform setSeed(long seed) {
		randomGenerator = new Random(seed);
		return this;
	}
	
	@Override
	public Uniform setMean(double mean) {
		double currentMean = (to+from)/2;
		from += mean-currentMean;
		to += mean-currentMean;
		return this;
	}
	
	@Override
	public Uniform setDeviation(double std) {
		double currentMean = (to+from)/2;
		double nextRange = std*sqrt12;
		from = currentMean - nextRange;
		to = currentMean + nextRange;
		return this;
	}
	
	@Override
	public double getMean() {
		return (to+from)/2;
	}
	
	@Override
	public double getDeviation() {
		return (from-to)/sqrt12;
	}
	
	@Override
	public double sample() {
		if(randomGenerator==null)
			randomGenerator = new Random();
		return from+randomGenerator.nextFloat()*(to-from);
	}
}
