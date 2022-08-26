package mklab.JGNN.nn.initializers;

import mklab.JGNN.core.Distribution;
import mklab.JGNN.core.distribution.Uniform;


/**
 * This is a {@link VariancePreservingInitializer}.
 * 
 * @author Emmanouil Krasanakis
 */
public class KaimingNormal extends VariancePreservingInitializer {
	@Override
	protected Distribution getDistribution(long rows, long cols, double gain) {
		double a = gain*Math.sqrt(3./rows);
		return new Uniform(-a, a);
	}
}
