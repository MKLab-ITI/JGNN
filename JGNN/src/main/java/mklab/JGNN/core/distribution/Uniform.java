package mklab.JGNN.core.distribution;

import mklab.JGNN.core.Distribution;

/**
 * Implements a Uniform {@link Distribution} of given bounds.
 * @author Emmanouil Krasanakis
 */
public class Uniform implements Distribution {
	private double from;
	private double to;
	public Uniform() {
		this(0, 1);
	}
	public Uniform(double from, double to) {
		this.from = from;
		this.to = to;
	}
	@Override
	public double sample() {
		return from+Math.random()*(to-from);
	}
}
