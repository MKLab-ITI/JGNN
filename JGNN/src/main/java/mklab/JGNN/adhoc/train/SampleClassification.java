package mklab.JGNN.adhoc.train;

import java.util.Arrays;
import java.util.List;

import mklab.JGNN.adhoc.BatchData;
import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.matrix.WrapRows;

/**
 * Extends the {@link ModelTraining} class to train {@link Model} instances from
 * feature and label matrices. This is a generic classification scheme that also
 * supports (and follows the training data flow of) traditional neural networks
 * that can produce batch predictions. If the model is a GNN, it is assumed that
 * it contains the graph adjacency matrix and the node features as constants,
 * and its input are node identifiers. This scheme is automated under
 * {@link mklab.JGNN.adhoc.parsers.FastBuilder#classify()}. In this case,
 * classification features should be an identity vertical matrix of node
 * identifiers; in the simplest case a vertical matrix organization of
 * [0,1,2,3,4,...]. Labels should be a matrix with predictions for corresponding
 * rows. Batch data generators create only one instance of data for each batch,
 * where the instance contains data for all trained or validated nodes.
 * 
 * @author Emmanouil Krasanakis
 * @see #train(Model)
 */
public class SampleClassification extends ModelTraining {
	private Matrix features;
	private Matrix labels;
	private Slice trainingSamples;
	private Slice validationSamples;

	/**
	 * Sets the feature matrix of data samples, where each row corresponds to a
	 * different sample. Features and labels should have the same number of rows.
	 * 
	 * @param features The feature matrix.
	 * @return <code>this</code> classification training instance.
	 */
	public SampleClassification setFeatures(Matrix features) {
		if (this.features != null)
			throw new RuntimeException("Can only set features once in a SampleClassification instance.");
		this.features = features;
		return this;
	}

	/**
	 * Sets the label matrix of data samples, where each row corresponds to a
	 * different sample. For example, each row may be an one-hot encoding of the
	 * respective sample's class when accessed. Features and labels should have the
	 * same number of rows.
	 * 
	 * @param labels The label matrix.
	 * @return <code>this</code> classification training instance.
	 */
	public SampleClassification setOutputs(Matrix labels) {
		if (this.labels != null)
			throw new RuntimeException("Can only set labels once in a SampleClassification instance.");
		this.labels = labels;
		return this;
	}

	/**
	 * Sets a slice of training samples. These should be identifiers of
	 * feature/label rows; basically, they reflect which rows of these matrices
	 * should be retrieved during training. If multiple batches are set, for example
	 * with {@link #setNumBatches(int)}, then these samples are further split for
	 * each batch.
	 * 
	 * @param trainingSamples The slice of training samples.
	 * @return <code>this</code> classification training instance.
	 */
	public SampleClassification setTrainingSamples(Slice trainingSamples) {
		if (this.trainingSamples != null)
			throw new RuntimeException("Can only set a training sample slice once in a SampleClassification instance.");
		this.trainingSamples = trainingSamples;
		return this;
	}

	/**
	 * Sets a slice of validation samples. These should be identifiers of
	 * feature/label rows; basically, they reflect which rows of these matrices
	 * should be retrieved during validation.
	 * 
	 * @param validationSamples The slice of validation samples.
	 * @return <code>this</code> classification training instance.
	 */
	public SampleClassification setValidationSamples(Slice validationSamples) {
		if (this.validationSamples != null)
			throw new RuntimeException(
					"Can only set a validation sample slice once in a SampleClassification instance.");
		this.validationSamples = validationSamples;
		return this;
	}

	@Override
	protected void onStartEpoch(int epoch) {
		if (stochasticGradientDescent)
			trainingSamples.shuffle(epoch);
	}

	@Override
	protected List<BatchData> getBatchData(int batch, int epoch) {
		if (features == null)
			throw new RuntimeException(
					"Cannot obtain batch data for SampleClassification without first setting sample features.");
		if (labels == null)
			throw new RuntimeException(
					"Cannot obtain batch data for SampleClassification without first setting sample labels.");
		if (trainingSamples == null)
			throw new RuntimeException(
					"Cannot obtain batch data for SampleClassification without first setting a training data slice.");
		if (validationSamples == null)
			throw new RuntimeException(
					"Cannot obtain batch data for SampleClassification without first setting a validation data slice.");
		int start = (trainingSamples.size() / numBatches) * batch;
		int end = Math.min(trainingSamples.size(), start + (trainingSamples.size() / numBatches));
		Matrix trainFeatures = new WrapRows(features.accessRows(trainingSamples.range(start, end)))
				.setDimensionName(features);
		Matrix trainLabels = new WrapRows(labels.accessRows(trainingSamples.range(start, end)))
				.setDimensionName(labels);
		return Arrays.asList(new BatchData(Arrays.asList(trainFeatures), Arrays.asList(trainLabels)));
	}

	@Override
	protected List<BatchData> getValidationData(int epoch) {
		Matrix trainFeatures = new WrapRows(features.accessRows(validationSamples))
				.setDimensionName(features.getRowName(), features.getColName());
		Matrix trainLabels = new WrapRows(labels.accessRows(validationSamples));// .setDimensionName(labels.getRowName(),
																				// labels.getColName());
		return Arrays.asList(new BatchData(Arrays.asList(trainFeatures), Arrays.asList(trainLabels)));
	}
}
