package mklab.JGNN.adhoc;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.ThreadPool;
import mklab.JGNN.nn.Loss;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.Optimizer;
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
public class ModelTraining {
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
	 * Set
	 * 
	 * @param loss
	 * @return
	 */
	public ModelTraining setLoss(Loss loss) {
		this.loss = loss;
		return this;
	}

	public ModelTraining setValidationLoss(Loss loss) {
		this.validationLoss = loss;
		return this;
	}

	/**
	 * Sets an {@link Optimizer} instance to controls parameter updates during
	 * training. If the provided optimizer is not an instance of
	 * {@link BatchOptimizer}, it is forcefully wrapped by the latter. Training
	 * calls the batch optimizer's update method after every batch.
	 * 
	 * @param optimizer The desired optimizer.
	 * @return <code>this</code> model training instance.
	 * @see #train(Model, Matrix, Matrix, Slice, Slice)
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
	 */
	public ModelTraining setNumBatches(int numBatches) {
		this.numBatches = numBatches;
		return this;
	}

	/**
	 * Sets whether the training strategy should reflect stochastic gradient descent
	 * by randomly sampling from the training dataset to obtain data samples. If
	 * <code>true</code>, both this feature and acceptable thread-based
	 * paralellization is enabled. Parallelization makes use of JGNN's
	 * {@link ThreadPool}.
	 * 
	 * @param paralellization A boolean value indicating whether this feature is
	 *                        enabled.
	 * @return <code>this</code> model training instance.
	 * @see #setNumBatches(int)
	 * @see #train(Model, Matrix, Matrix, Slice, Slice)
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
	 * This is a leftover method from an earlier version of JGNN's interface.
	 * 
	 * @deprecated This method has been moved to
	 *             {@link mklab.JGNN.adhoc.train.NodeClassification#train(Model, Matrix, Matrix, Slice, Slice)}
	 */
	public Model train(Model model, Matrix features, Matrix labels, Slice trainingSamples, Slice validationSamples) {
		throw new RuntimeException(
				"The ModelTraining.train method has been moved to NodeClassification.train since version 1.3.28. "
						+ "\n    For valid code, create a NodeClassification instance instead of a ModelTraining instance. "
						+ "\n    This method may be made abstract or removed completely in future versions, and will probably be replaced"
						+ "\n    with a uniform interface for any predictive task.");
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
}
