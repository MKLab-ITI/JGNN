package mklab.JGNN.nn.initializers;

import mklab.JGNN.core.Distribution;
import mklab.JGNN.core.distribution.Uniform;


/**
 * This is a {@link VariancePreservingInitializer}.
 * 
 * @author Emmanouil Krasanakis
 */
public class XavierUniform extends VariancePreservingInitializer {
	@Override
	protected Distribution getDistribution(long rows, long cols, double gain) {
		double a = gain*Math.sqrt(6./(rows + cols));
		return new Uniform(-a, a);
	}
}
