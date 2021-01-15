package mklab.JGNN.models.measures;

import java.util.List;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.models.Measure;

public class Accuracy extends Measure {
	@Override
	public double evaluate(Tensor predictions, Tensor labels) {
		if(labels.size()!=predictions.size())
			throw new RuntimeException("Predictions should have the same size as labels");
		long t = 0;
		for(long idx : predictions.getNonZeroElements()) {
			double isP = predictions.get(idx)>0.5?1:0;
			if(isP == labels.get(idx))
				t += 1;
		}
		return t / (double)predictions.size();
	}
	@Override
	public final double evaluate(List<Tensor> predictions, List<Tensor> labels) {
		if(predictions.size()==1)
			return evaluate(predictions.get(0), labels.get(0));
		double average = 0;
		for(int i=0;i<predictions.get(0).size();i++)
			average += argmax(predictions, i)==argmax(labels, i)?1:0;
		return average / predictions.get(0).size();
		
	}
	private static int argmax(List<Tensor> tensors, long row) {
		double maxValue = Double.NEGATIVE_INFINITY;
		int argmax = -1;
		for(int i=0;i<tensors.size();i++)
			if(maxValue<tensors.get(i).get(row)) {
				maxValue = tensors.get(i).get(row);
				argmax = i;
			}
		return argmax;
				
	}
}
