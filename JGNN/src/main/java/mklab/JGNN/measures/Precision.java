package mklab.JGNN.measures;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.data.Measure;

public class Precision extends Measure {
	@Override
	public double evaluate(Tensor predictions, Tensor labels) {
		if(labels.size()!=predictions.size())
			throw new RuntimeException("Predictions should have the same size as labels");
		long tp = 0;
		long p = 0;
		for(long idx : predictions.getNonZeroElements()) {
			double isP = predictions.get(idx)>0.5?1:0;
			if(isP==1 && labels.get(idx)==1)
				tp += 1;
			if(isP==1)
				p += 1;
		}
		return tp / (double)p;
	}
}
