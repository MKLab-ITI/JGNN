package mklab.JGNN.models.measures;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.models.Measure;

public class RMSE extends Measure {
	@Override
	public double evaluate(Tensor predictions, Tensor labels) {
		if(labels.size()!=predictions.size())
			throw new RuntimeException("Predictions should have the same size as labels");
		double se = 0;
		for(long idx : predictions.getNonZeroElements()) {
			double diff = predictions.get(idx)-labels.get(idx);
			se += diff*diff;
		}
		return Math.sqrt(se / predictions.size());
	}
}
