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
	private Random randomGenerator = new Random();
	
	public Normal() {
		this(0, 1);
	}
		
	public Normal(double mean, double std) {
		this.mean = mean;
		this.std = std;
	}
	
	@Override
	public double sample() {
		return randomGenerator.nextGaussian()*std + mean;
	}

}
