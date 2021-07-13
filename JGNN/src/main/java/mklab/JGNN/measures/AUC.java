package mklab.JGNN.measures;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Sort;
import mklab.JGNN.data.Measure;

public class AUC extends Measure {
	@Override
	public double evaluate(Tensor predictions, Tensor labels) {
		if(labels.size()!=predictions.size())
			throw new RuntimeException("Predictions should have the same size as labels");
		long rank = 0;
		long positiveRankSum = 0;
		long n1 = 0;
		for(long idx : Sort.sortedIndexes(predictions.toArray())) {
			rank += 1;
			if(labels.get(idx)!=0) {
				positiveRankSum += rank;
				n1 += 1;
			}
		}
		positiveRankSum -= n1*(n1+1)/2;
		long n2 = predictions.size()-n1;
		return positiveRankSum/(double)n1/n2;
	}
}
