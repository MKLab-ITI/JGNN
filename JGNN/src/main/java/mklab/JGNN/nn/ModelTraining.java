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
	public ModelTraining setVerbose(boolean verbose) {
		this.verbose = verbose;
		return this;
	}
	public ModelTraining setLoss(Loss loss) {
		this.loss = loss;
		return this;
	}
	public ModelTraining setValidationLoss(Loss loss) {
		this.validationLoss = loss;
		return this;
	}
	public ModelTraining setOptimizer(Optimizer optimizer) {
		this.optimizer = new BatchOptimizer(optimizer);
		return this;
	}
	public ModelTraining setNumBatches(int numBatches) {
		this.numBatches = numBatches;
		return this;
	}
	public ModelTraining setParallelizedStochasticGradientDescent(boolean paralellization) {
		this.paralellization = paralellization;
		this.stochasticGradientDescent = paralellization;
		return this;
	}
	public ModelTraining setEpochs(int epochs) {
		this.epochs = epochs;
		return this;
	}
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
		setEpochs(modelBuilder.getConfigOrDefault("epochs", epochs));
		numBatches = modelBuilder.getConfigOrDefault("batches", numBatches);
		setPatience(modelBuilder.getConfigOrDefault("patience", patience));
		return this;
	}
}
