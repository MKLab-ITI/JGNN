package mklab.JGNN.adhoc;

import java.util.HashMap;
import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Memory;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.ThreadPool;
import mklab.JGNN.nn.Loss;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.Optimizer;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;

/**
 * This is a helper class that automates the definition of training processes of
 * {@link Model} instances by defining the number of epochs, loss functions,
 * number of batches and the ability to use {@link ThreadPool} for parallelized
 * batch computations.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class ModelTraining {
	protected BatchOptimizer optimizer;
	protected int numBatches = 1;
	protected int epochs = 300;
	protected int patience = Integer.MAX_VALUE;
	protected boolean paralellization = false;
	protected boolean stochasticGradientDescent = false;
	protected Loss loss, validationLoss;
	protected boolean verbose = false;

	public ModelTraining() {
	}

	/**
	 * @param verbose Whether an error message will be printed.
	 * @return The model training instance.
	 * @deprecated This method was available in earlier JGNN versions but will be
	 *             gradually phased out. Instead, wrap the validation loss within
	 *             {@link mklab.JGNN.nn.loss.report.VerboseLoss} to replicate the
	 *             same behavior.
	 */
	public ModelTraining setVerbose(boolean verbose) {
		System.err.println("WARNING: The setVerbose method was available in earlier JGNN versions"
				+ "\n    but will be gradually phased out. Instead, wrap the validation"
				+ "\n    loss within a VerboseLoss instance to replicate the same"
				+ "\n    behavior. Look for more losses of the mklab.JGNN.nn.loss.report"
				+ "\n    package for more types of training feedback.");
		this.verbose = verbose;
		return this;
	}

	/**
	 * Sets which {@link mklab.JGNN.nn.Loss} should be applied on training batches
	 * (the loss is averaged across batches, but is aggregated as a sum within each
	 * batch by {@link BatchOptimizer}). Model training mainly uses the loss's
	 * {@link mklab.JGNN.nn.Loss#derivative(Tensor, Tensor)} method, alongside
	 * {@link mklab.JGNN.nn.Loss#onEndEpoch()} and
	 * {@link mklab.JGNN.nn.Loss#onEndTraining()}. If no validation loss is set, in
	 * which case the training loss is also used for validation.
	 * 
	 * @param loss The loss's instance.
	 * @return The model training instance.
	 * @see #setValidationLoss(Loss)
	 */
	public ModelTraining setLoss(Loss loss) {
		this.loss = loss;
		return this;
	}

	/**
	 * Sets which {@link mklab.JGNN.nn.Loss} should be applied on validation data on
	 * each epoch. The loss's {@link mklab.JGNN.nn.Loss#onEndEpoch()},
	 * {@link mklab.JGNN.nn.Loss#onEndTraining()}, and
	 * {@link mklab.JGNN.nn.Loss#evaluate(Tensor, Tensor)} methods are used. In the
	 * case where validation is split into multiple instances of batch data, which
	 * may be necessary for complex scenarios like graph classification, the loss
	 * value is averaged across those batches. The methods mentioned above are not
	 * used by losses employed in training.
	 * 
	 * @param loss The loss's instance.
	 * @return The model training instance.
	 * @see #setLoss(Loss)
	 */
	public ModelTraining setValidationLoss(Loss loss) {
		this.validationLoss = loss;
		return this;
	}

	/**
	 * Sets an {@link Optimizer} instance to controls parameter updates during
	 * training. If the provided optimizer is not an instance of
	 * {@link BatchOptimizer}, it is forcefully wrapped by the latter. Training
	 * calls the batch optimizer's update method after every batch. Each batch could
	 * contain multiple instances of batch data. However, the total number of
	 * applied gradient updates is always equal to the value set by
	 * {@link #setNumBatches(int)}.
	 * 
	 * @param optimizer The desired optimizer.
	 * @return <code>this</code> model training instance.
	 * @see #train(Model)
	 */
	public ModelTraining setOptimizer(Optimizer optimizer) {
		if (optimizer instanceof BatchOptimizer)
			this.optimizer = (BatchOptimizer) optimizer;
		else
			this.optimizer = new BatchOptimizer(optimizer);
		return this;
	}

	/**
	 * Sets the number of batches training data slices should be split into.
	 * 
	 * @param numBatches The desired number of batches. Default is 1.
	 * @return <code>this</code> model training instance.
	 * @see #setParallelizedStochasticGradientDescent(boolean)
	 * @see #setNumParallellBatches(int)
	 */
	public ModelTraining setNumBatches(int numBatches) {
		this.numBatches = numBatches;
		return this;
	}

	/**
	 * Sets whether the training strategy should reflect stochastic gradient descent
	 * by randomly sampling from the training data samples. If <code>true</code>,
	 * both this feature and acceptable thread-based paralellization is enabled.
	 * Parallelization uses JGNN's {@link ThreadPool}.
	 * 
	 * @param paralellization A boolean value indicating whether this feature is
	 *                        enabled.
	 * @return <code>this</code> model training instance.
	 * @see #setNumBatches(int)
	 * @see #train(Model, Matrix, Matrix, Slice, Slice)
	 * @see #setNumParallellBatches(int)
	 */
	public ModelTraining setParallelizedStochasticGradientDescent(boolean paralellization) {
		this.paralellization = paralellization;
		this.stochasticGradientDescent = paralellization;
		return this;
	}

	/**
	 * Sets the maximum number of epochs for which training runs. If no patience has
	 * been set, training runs for exactly this number of epochs.
	 * 
	 * @param epochs The maximum number of epochs.
	 * @return <code>this</code> model training instance.
	 * @see #setPatience(int)
	 */
	public ModelTraining setEpochs(int epochs) {
		this.epochs = epochs;
		return this;
	}

	/**
	 * Sets the patience of the training strategy that performs early stopping. If
	 * training does not encounter a smaller validation loss for this number of
	 * epochs, it stops.
	 * 
	 * @param patience The number of patience epochs. Default is Integer.MAX_VALUE
	 *                 to effectively disable this feature and let training always
	 *                 reach the maximum number of set epochs.
	 * @return <code>this</code> model training instance.
	 * @see #setEpochs(int)
	 */
	public ModelTraining setPatience(int patience) {
		this.patience = patience;
		return this;
	}

	/**
	 * This is a leftover method from an earlier version of JGNN's interface. For
	 * the time being, there is no good alternative, but it will be phased out.
	 * 
	 * @deprecated This method's full implementation has been moved to
	 *             {@link #train(Model)}
	 */
	public Model train(Model model, Matrix features, Matrix labels, Slice trainingSamples, Slice validationSamples) {
		throw new RuntimeException(
				"The ModelTraining.train method with more than one arguments is deprecated since version 1.3.28."
						+ "\n    For valid code, create a NodeClassification instance instead of a ModelTraining instance, "
						+ "\n    set its data, and call the method ModelTraining.train(Model) instead.");
	}

	/**
	 * Performs necessary training operations at the beginning of each epoch. These
	 * typically consist of dataset shuffling if
	 * {@link #setParallelizedStochasticGradientDescent(boolean)} is enabled.
	 * 
	 * @param epoch The epoch that now starts. Takes values 0,1,2,...,epochs-1,
	 *              though early stopping may not reach the maximum number.
	 */
	protected abstract void onStartEpoch(int epoch);

	/**
	 * Performs any cleanup operations at the end of the {@link #train(Model)} loop.
	 * This method is mostly used to "unlock" data insertions to the training
	 * process.
	 */
	protected void onEndTraining() {
	}

	/**
	 * Returns a list {@link BatchData} instance to be used for a specific batch and
	 * training epoch. This list may have only one entry if the whole batch can be
	 * organized into one pair of model inputs-outputs (e.g., in node
	 * classification). This method is overloaded by classes extending
	 * {@link ModelTraining} to let them work as dataset loaders. Batch data may be
	 * created anew each time, though they are often transparent views of parts of
	 * training data. Batch data generation may be parallelized, depending on the
	 * whether {@link #setParallelizedStochasticGradientDescent(boolean)} is
	 * enabled. If some operations (e.g., data shuffling) take place at the
	 * beginning of each epoch, they instead reside in the {@link #startEpoch()}
	 * method.
	 * 
	 * @param batch The batch identifier. Takes values 0,1,2,..,numBatches-1.
	 * @param epoch The epoch in which the batch is extracted. Takes values
	 *              0,1,2,...,epochs-1, though early stopping may not reach the
	 *              maximum number.
	 * @return An list of batch data instances.
	 */
	protected abstract List<BatchData> getBatchData(int batch, int epoch);

	/**
	 * Returns a {@link BatchData} instance to be used for validation at a given
	 * training epoch. This list may have only one entry if the whole batch can be
	 * organized into one pair of model inputs-outputs (e.g., in node
	 * classification). This method is overloaded by classes extending
	 * {@link ModelTraining} to let them work as dataset loaders. Batch data may be
	 * created anew each time, though they are often transparent views of parts of
	 * training data.
	 * 
	 * @param epoch The epoch in which the batch is extracted. Takes values
	 *              0,1,2,...,epochs-1, though early stopping may not reach the
	 *              maximum number.
	 * @return An list of batch data instances.
	 */
	protected abstract List<BatchData> getValidationData(int epoch);

	/**
	 * Trains the parameters of a {@link Model} based on current settings and the
	 * data.
	 * 
	 * @param model The model instance to train.
	 */
	public Model train(Model model) {
		double minLoss = Double.POSITIVE_INFINITY;
		HashMap<Parameter, Tensor> minLossParameters = new HashMap<Parameter, Tensor>();
		int currentPatience = patience;
		Loss validLoss = validationLoss != null ? validationLoss : loss;
		for (int epoch = 0; epoch < epochs; epoch++) {
			onStartEpoch(epoch);
			int epochId = epoch;
			for (int batch = 0; batch < numBatches; batch++) {
				int batchId = batch;
				Runnable batchCode = new Runnable() {
					@Override
					public void run() {
						for (BatchData batchData : getBatchData(batchId, epochId)) 
							model.train(loss, optimizer, batchData.getInputs(), batchData.getOutputs());
						if (stochasticGradientDescent)
							optimizer.updateAll();
					}
				};
				if (paralellization)
					ThreadPool.getInstance().submit(batchCode);
				else
					batchCode.run();
				// System.out.println(System.currentTimeMillis()-tic);
			}
			if (paralellization)
				ThreadPool.getInstance().waitForConclusion();
			if (!stochasticGradientDescent)
				optimizer.updateAll();
			loss.onEndEpoch();
			
			Memory.scope().enter();
			double totalLoss = 0;
			List<BatchData> allValidationData = getValidationData(epoch);
			for (BatchData validationData : allValidationData) {
				List<Tensor> outputs = model.predict(validationData.getInputs());
				totalLoss += validLoss.evaluate(outputs.get(0), validationData.getOutputs().get(0));
			}
			Memory.scope().exit();
			if (totalLoss != 0)
				totalLoss /= allValidationData.size();

			if (totalLoss < minLoss) {
				currentPatience = patience;
				minLoss = totalLoss;
				for (Parameter parameter : model.getParameters())
					minLossParameters.put(parameter, parameter.get().copy());
			}

			if (verbose)
				System.out.println("Epoch " + epoch + " with loss " + totalLoss);
			validLoss.onEndEpoch();
			currentPatience -= 1;
			if (currentPatience == 0)
				break;
		}
		for (Parameter parameter : model.getParameters())
			parameter.set(minLossParameters.get(parameter));
		loss.onEndTraining();
		validLoss.onEndTraining();
		onEndTraining();
		return model;
	}

	/**
	 * Retrieves the learning rate (lr), epochs, batches, and patience parameters
	 * from the configurations of a
	 * 
	 * @param modelBuilder
	 * @return <code>this</code> model training instance.
	 */
	public ModelTraining configFrom(ModelBuilder modelBuilder) {
		setOptimizer(new Adam(modelBuilder.getConfigOrDefault("lr", 0.01)));
		setEpochs((int) modelBuilder.getConfigOrDefault("epochs", epochs));
		numBatches = (int) modelBuilder.getConfigOrDefault("batches", numBatches);
		setPatience((int) modelBuilder.getConfigOrDefault("patience", patience));
		return this;
	}
	
	/**
	 * Sets the number of batches for each training epoch and sets the training
	 * to either be executed in one thread if there is only one batch or in
	 * multiple threads of the {@link ThreadPool} if there are multiple
	 * batches. This is a shorthand for invoking related methods sequentially
	 * and without forgetting to set batches when enabling parallelization.
	 * @param numBatches The number of batches in each training epoch.
	 * 
	 * @return <code>this</code> model training instance.
	 * @see #setNumBatches(int)
	 * @see #setParallelizedStochasticGradientDescent(boolean)
	 */
	public ModelTraining setNumParallellBatches(int numBatches) {
		setNumBatches(numBatches);
		setParallelizedStochasticGradientDescent(numBatches!=1);
		return this;
	}
}
