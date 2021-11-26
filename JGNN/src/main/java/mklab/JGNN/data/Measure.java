package mklab.JGNN.data;

import java.util.List;

import mklab.JGNN.core.Tensor;

public abstract class Measure {
	public double evaluate(List<Tensor> predictions, List<Tensor> labels) {
		if(predictions==null || labels==null)
			throw new IllegalArgumentException();
		if(predictions.isEmpty())
			throw new IllegalArgumentException();
		if(predictions.size()!=labels.size())
			throw new IllegalArgumentException();
		double average = 0;
		for(int i=0;i<predictions.size();i++)
			average += evaluate(predictions.get(i), labels.get(i));
		return average / predictions.size();
	}
	public abstract double evaluate(Tensor predictions, Tensor labels);
}
