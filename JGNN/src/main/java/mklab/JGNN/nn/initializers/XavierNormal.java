package mklab.JGNN.nn.initializers;

import mklab.JGNN.core.Distribution;
import mklab.JGNN.core.distribution.Normal;


/**
 * This is a {@link VariancePreservingInitializer}.
 * 
 * @author Emmanouil Krasanakis
 */
public class XavierNormal extends VariancePreservingInitializer {
	@Override
	protected Distribution getDistribution(long rows, long cols, double gain) {
		return new Normal(0, gain*Math.sqrt(2./(rows + cols)));
	}
}
