package mklab.JGNN.core.distribution;

import java.util.Random;

import mklab.JGNN.core.Distribution;

/**
 * Implements a Normal {@link Distribution} of given mean and standard deviation.
 * @author Emmanouil Krasanakis
 */
public class Normal implements Distribution {
	private double mean;
	private double std;
	private Random randomGenerator;
	
	public Normal() {
		this(0, 1);
	}
		
	public Normal(double mean, double std) {
		this.mean = mean;
		this.std = std;
		randomGenerator = null;
	}
	
	@Override
	public Normal setSeed(long seed) {
		randomGenerator = new Random(seed);
		return this;
	}
	
	@Override
	public Normal setMean(double mean) {
		this.mean = mean;
		return this;
	}
	
	@Override
	public Normal setDeviation(double std) {
		this.std = std;
		return this;
	}
	
	@Override
	public double getMean() {
		return mean;
	}
	@Override
	public double getDeviation() {
		return std;
	}
	
	@Override
	public double sample() {
		if(randomGenerator==null) 
			randomGenerator = new Random();
		return randomGenerator.nextGaussian()*std + mean;
	}

}
