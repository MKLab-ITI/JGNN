package mklab.JGNN.adhoc.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mklab.JGNN.adhoc.BatchData;
import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * Extends the {@link ModelTraining} class to be able to train
 * {@link mklab.JGNN.nn.Model} instances for attributed graph functions (AGFs).
 * Training needs to account for a list of graphs, corresponding graph node
 * features, and corresponding graph labels. Each label holds the one-hot
 * encoding of each graph. Fill data with the method
 * {@link #addGraph(Matrix, Matrix, Tensor)}.
 * 
 * @author Emmanouil Krasanakis
 */
public class AGFTraining extends ModelTraining {
	private List<Matrix> graphs = new ArrayList<Matrix>();
	private List<Matrix> nodeFeatures = new ArrayList<Matrix>();
	private List<Tensor> graphLabels = new ArrayList<Tensor>();
	private Slice trainingSamples;
	private Slice validationSamples;

	public AGFTraining setGraphs(List<Matrix> graphs) {
		this.graphs = graphs;
		return this;
	}

	public AGFTraining setNodeFeatures(List<Matrix> nodeFeatures) {
		this.nodeFeatures = nodeFeatures;
		return this;
	}

	public AGFTraining setGraphLabels(List<Tensor> graphLabels) {
		this.graphLabels = graphLabels;
		return this;
	}

	public AGFTraining addGraph(Matrix graph, Matrix features, Tensor labels) {
		if (trainingSamples != null)
			throw new RuntimeException("Cannot add data to GraphClassification after setting a validation split.");
		graphs.add(graph);
		nodeFeatures.add(features);
		graphLabels.add(labels);
		return this;
	}

	public AGFTraining setValidationSplit(double validationFraction) {
		Slice indices = new Slice(new Range(0, graphs.size()));
		indices.shuffle();
		validationSamples = indices.range(0, validationFraction);
		trainingSamples = indices.range(validationFraction, 1);
		return this;
	}

	@Override
	protected void onStartEpoch(int epoch) {
		if (trainingSamples == null)
			throw new RuntimeException(
					"Need to create a train-validation split on Graph clasification data before training starts.");
		if (stochasticGradientDescent)
			trainingSamples.shuffle(epoch);
	}

	@Override
	protected void onEndTraining() {
		trainingSamples = null; // basically unlocks data insertion again
		validationSamples = null;
	}

	@Override
	protected List<BatchData> getBatchData(int batch, int epoch) {
		int start = (trainingSamples.size() / numBatches) * batch;
		int end = Math.min(trainingSamples.size(), start + (trainingSamples.size() / numBatches));
		List<BatchData> batchData = new ArrayList<BatchData>(end - start);
		for (long i : trainingSamples.range(start, end))
			batchData.add(new BatchData(Arrays.asList(nodeFeatures.get((int) i), graphs.get((int) i)),
					Arrays.asList(graphLabels.get((int) i))));
		return batchData;
	}

	@Override
	protected List<BatchData> getValidationData(int epoch) {
		List<BatchData> batchData = new ArrayList<BatchData>(validationSamples.size());
		for (long i : validationSamples)
			batchData.add(new BatchData(Arrays.asList(nodeFeatures.get((int) i), graphs.get((int) i)),
					Arrays.asList(graphLabels.get((int) i))));
		return batchData;
	}

}
