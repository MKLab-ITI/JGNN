package mklab.JGNN.models.relational;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.AccessSubtensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.data.TrainingData;

public class RelationalData implements TrainingData {
	private Tensor edgeSrc;
	private Tensor edgeDst;
	private Tensor edgeLabels;
	public static enum NegativeSamplingType {RANDOM, PERMUTATION, NEIGHBOR_PERMUTATION};
	
	public RelationalData(SparseMatrix W, NegativeSamplingType samplingType) {
		this(W, W, samplingType);
	}
	
	public RelationalData(SparseMatrix W, Matrix avoidNegativeEdges, NegativeSamplingType samplingType) {
		long numEdges = 0;
		for(Entry<Long, Long> edge : W.getNonZeroEntries())
			if(edge.getKey()!=edge.getValue())
				numEdges += 2;
		edgeSrc = new DenseTensor(numEdges);
		edgeDst = new DenseTensor(numEdges);
		edgeLabels = new DenseTensor(numEdges);
		fillTrainingData(W, avoidNegativeEdges, edgeSrc, edgeDst, edgeLabels, samplingType);
	}
	
	@Override
	public List<Tensor> getInputs() {
		return Arrays.asList(edgeSrc, edgeDst);
	}

	@Override
	public List<Tensor> getOutputs() {
		return Arrays.asList(edgeLabels);
	}
	
	protected void fillTrainingData(Matrix W, Matrix avoidNegativeEdges, Tensor uList, Tensor vList, Tensor labels, NegativeSamplingType samplingType) {
		int pos = 0;
		for(Entry<Long, Long> edge : W.getNonZeroEntries()) {
			int u = (int)(long)edge.getKey();
			int v = (int)(long)edge.getValue();
			if(u==v)
				continue;
			uList.put(pos, u);
			vList.put(pos, v);
			labels.put(pos, 1);
			pos += 1;
			
			int retries = 1000;
			int negu = -1;
			int negv = -1;
			while(negu==negv || W.get(negu, negv)!=0 || avoidNegativeEdges.get(negu, negv)!=0) {
				if(samplingType==NegativeSamplingType.RANDOM) {
					negu = (int)(Math.random()*W.getRows());
					negv = (int)(Math.random()*W.getRows());
				}
				else if(samplingType==NegativeSamplingType.PERMUTATION) {
					negu = (int)(Math.random()*W.getRows());
					negv = (int)(Math.random()*W.getRows());
					if(Math.random()<0.5)
						negu = u;
					else
						negv = v;
				}
				else
					throw new RuntimeException("Sampling method not implemented yet "+samplingType);
				retries -= 1;
				if(retries==0)
					throw new RuntimeException("Randomly sampling 1000 time did not yield valid negative samples. Consider changing sampling strategy.");
			}
			uList.put(pos, negu);
			vList.put(pos, negv);
			pos += 1;
		}
	}
}
