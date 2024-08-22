package mklab.JGNN.nn;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Memory;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.ThreadPool;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.nn.inputs.Parameter;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;

/**
 * This is a helper class that automates the definition of training processes of {@link Model} instances 
 * by defining the number of epochs, loss functions, number of batches and the ability to use {@link ThreadPool} 
 * for parallelized batch computations.
 * 
 * @author Emmanouil Krasanakis
 */
public class ModelTraining {
	private BatchOptimizer optimizer;
	private int numBatches = 1;
	private int epochs = 300;
	private int patience = Integer.MAX_VALUE;
	private boolean paralellization = false;
	private boolean stochasticGradientDescent = false;
	private Loss loss, validationLoss;
	private boolean verbose = false;
	
	public ModelTraining() {
	}
	
	/**
	 * @param verbose Whether an error message will be printed.
	 * @deprecated This method was available in earlier JGNN versions but will be gradually phased out.
	 * Instead, wrap the validation loss within {@link mklab.JGNN.nn.loss.report.VerboseLoss} to replicate
	 * the same behavior.
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
	 * Sets an {@link Optimizer} instance to controls parameter updates during training.
	 * If the provided optimizer is not an instance of {@link BatchOptimizer},
	 * it is forcefully wrapped by the latter. Training calls the batch optimizer's
	 * update method after every batch.
	 * @param optimizer The desired optimizer.
	 * @return <code>this</code> model training instance.
	 * @see #train(Model, Matrix, Matrix, Slice, Slice)
	 */
	public ModelTraining setOptimizer(Optimizer optimizer) {
		if(optimizer instanceof BatchOptimizer)
			this.optimizer = (BatchOptimizer) optimizer;
		else
			this.optimizer = new BatchOptimizer(optimizer);
		return this;
	}
	
	/**
	 * Sets the number of batches training data slices should be split into.
	 * @param numBatches The desired number of batches. Default is 1.
	 * @return <code>this</code> model training instance.
	 * @see #setParallelizedStochasticGradientDescent(boolean)
	 */
	public ModelTraining setNumBatches(int numBatches) {
		this.numBatches = numBatches;
		return this;
	}
	
	/**
	 * Sets whether the training strategy should reflect stochastic
	 * gradient descent by randomly sampling from the training dataset to obtain data samples.
	 * If <code>true</code>, both this feature and acceptable thread-based paralellization
	 * is enabled. Parallelization makes use of JGNN's {@link ThreadPool}.
	 * @param paralellization A boolean value indicating whether this feature is enabled.
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
	 * Sets the maximum number of epochs for which training runs. 
	 * If no patience has been set, training runs for exactly this
	 * number of epochs.
	 * @param epochs The maximum number of epochs.
	 * @return <code>this</code> model training instance.
	 * @see #setPatience(int)
	 */
	public ModelTraining setEpochs(int epochs) {
		this.epochs = epochs;
		return this;
	}
	
	/**
	 * Sets the patience of the training strategy that performs early stopping.
	 * If training does not encounter a smaller validation loss for this number of 
	 * epochs, it stops. 
	 * @param patience The number of patience epochs. Default is Integer.MAX_VALUE to effectively disable this
	 * 	feature and let training always reach the maximum number of set epochs.
	 * @return <code>this</code> model training instance.
	 * @see #setEpochs(int)
	 */
	public ModelTraining setPatience(int patience) {
		this.patience = patience;
		return this;
	}
	
	/**
	 * Trains a {@link Model} instance based on current settings.
	 * 
	 * @param model The model instance to train.
	 * @param features A matrix whose columns correspond to sample features.
	 * @param labels A matrix whose columns correspond to sample (one hot) labels.
	 * @param trainingSamples Which columns to select for training.
	 * @return The trained <code>model</code> (the same instance as the first argument).
	 */
	public Model train(Model model, 
			Matrix features, 
			Matrix labels, 
			Slice trainingSamples,
			Slice validationSamples) {
		// ACTUΑL TRAINING
		double minLoss = Double.POSITIVE_INFINITY;
		HashMap<Parameter, Tensor> minLossParameters = new HashMap<Parameter, Tensor>();
		int currentPatience = patience;
		for(int epoch=0;epoch<epochs;epoch++) {
			//long tic = System.currentTimeMillis();
			if(!stochasticGradientDescent)
				trainingSamples.shuffle(epoch);
			double[] batchLosses = new double[numBatches];
			for(int batch=0;batch<numBatches;batch++) {
				if(stochasticGradientDescent)
					trainingSamples.shuffle(epoch);
				int start = (trainingSamples.size() / numBatches)*batch;
				int end = Math.min(trainingSamples.size(), start+(trainingSamples.size() / numBatches));
				int batchId = batch;
				Matrix trainFeatures = new WrapRows(features.accessRows(trainingSamples.range(start, end)))
						.setDimensionName(features.getRowName(), features.getColName());
				Matrix trainLabels = new WrapRows(labels.accessRows(trainingSamples.range(start, end)));
						//.setDimensionName(labels.getRowName(), labels.getColName());
				//System.out.println(System.currentTimeMillis()-tic);
				Runnable batchCode = new Runnable() {
					@Override
					public void run() {
						List<Tensor> outputs;
						outputs = model.train(loss, optimizer, Arrays.asList(trainFeatures), Arrays.asList(trainLabels));
						if(stochasticGradientDescent)
							optimizer.updateAll();
						if(validationSamples==null)
							batchLosses[batchId] = loss.evaluate(outputs.get(0), trainLabels);
					}
				};
				if(paralellization)
					ThreadPool.getInstance().submit(batchCode);
				else
					batchCode.run();
				//System.out.println(System.currentTimeMillis()-tic);
			}
			if(paralellization)
				ThreadPool.getInstance().waitForConclusion();
			if(!stochasticGradientDescent)
				optimizer.updateAll();
			double totalLoss = 0;
			if(validationSamples==null)
				for(double batchLoss : batchLosses)
					totalLoss += batchLoss/numBatches;
			else {
				Memory.scope().enter();
				Matrix validationFeatures = new WrapRows(features.accessRows(validationSamples));
				Matrix validationLabels = new WrapRows(labels.accessRows(validationSamples));
				List<Tensor> outputs = model.predict(Arrays.asList(validationFeatures));
				totalLoss = (validationLoss!=null?validationLoss:loss).evaluate(outputs.get(0), validationLabels);// outputs.get(0).multiply(-1).cast(Matrix.class).selfAdd(validationLabels).selfAbs().norm();
				Memory.scope().exit();
				//for(long i=0;i<validationLabels.getRows();i++)
				//	totalLoss -=(outputs.get(0).cast(Matrix.class).accessRow(i).argmax()==validationLabels.accessRow(i).argmax())?1:0;
				//totalLoss /= validationLabels.getRows();
			}
			if(totalLoss<minLoss) {
				currentPatience = patience;
				minLoss = totalLoss;
				for(Parameter parameter : model.getParameters()) 
					minLossParameters.put(parameter, parameter.get().copy());
			}
			if(verbose)
				System.out.println("Epoch "+epoch+" with loss "+totalLoss);
			currentPatience -= 1;
			if(currentPatience==0)
				break;
		}
		for(Parameter parameter : model.getParameters()) 
			parameter.set(minLossParameters.get(parameter));
		return model;
	}
	public ModelTraining configFrom(ModelBuilder modelBuilder) {
		setOptimizer(new Adam(modelBuilder.getConfigOrDefault("lr", 0.01)));
		setEpochs((int)modelBuilder.getConfigOrDefault("epochs", epochs));
		numBatches = (int)modelBuilder.getConfigOrDefault("batches", numBatches);
		setPatience((int)modelBuilder.getConfigOrDefault("patience", patience));
		return this;
	}
}
