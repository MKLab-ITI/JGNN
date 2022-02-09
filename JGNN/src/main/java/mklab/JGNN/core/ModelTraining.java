package mklab.JGNN.core;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.nn.optimizers.BatchOptimizer;

/**
 * This is a helper class that automates the definition of training processes of {@link Model} instances 
 * by defining the number of epochs, loss functions, number of batches and the ability to use {@link ThreadPool} 
 * for parallelized batch computations.
 * 
 * @author Emmanouil Krasanakis
 */
public class ModelTraining {
	public enum Loss {L2, CrossEntropy};
	private BatchOptimizer optimizer;
	private int numBatches = 1;
	private int epochs = 150;
	private boolean paralellization = true;
	private Loss loss;
	
	public ModelTraining() {
	}
	public ModelTraining setLoss(Loss loss) {
		this.loss = loss;
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
	public ModelTraining setParallelization(boolean paralellization) {
		this.paralellization = paralellization;
		return this;
	}
	public ModelTraining setEpochs(int epochs) {
		this.epochs = epochs;
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
	public Model train(Model model, Matrix features, Matrix labels, List<Long> trainingSamples) {
		for(int epoch=0;epoch<epochs;epoch++) {
			Collections.shuffle(trainingSamples, new Random(epoch));
			double[] batchLosses = new double[numBatches];
			for(int batch=0;batch<numBatches;batch++) {
				int start = (trainingSamples.size() / numBatches)*batch;
				int end = Math.min(trainingSamples.size(), start+(trainingSamples.size() / numBatches));
				int batchId = batch;
				Matrix trainFeatures = new WrapRows(features.accessRows(trainingSamples.subList(start, end)));
				Matrix trainLabels = new WrapRows(labels.accessRows(trainingSamples.subList(start, end)));
				Runnable batchCode = new Runnable() {
					@Override
					public void run() {
						List<Tensor> outputs;
						if(loss==Loss.L2)
							outputs = model.trainL2(optimizer, Arrays.asList(trainFeatures), Arrays.asList(trainLabels));
						else if(loss==Loss.CrossEntropy)
							outputs = model.trainCrossEntropy(optimizer, Arrays.asList(trainFeatures), Arrays.asList(trainLabels));
						else
							throw new RuntimeException("Unimplemented loss method "+loss);
						optimizer.updateAll();
						batchLosses[batchId] = outputs.get(0).multiply(-1).cast(Matrix.class).selfAdd(trainLabels).selfAbs().norm();
					}
				};
				if(paralellization)
					ThreadPool.getInstance().submit(batchCode);
				else
					batchCode.run();
			}
			if(paralellization)
				ThreadPool.getInstance().waitForConclusion();
			double totalLoss = 0;
			for(double batchLoss : batchLosses)
				totalLoss += batchLoss/numBatches;
			System.out.println("Epoch "+epoch+" with avg loss "+totalLoss);
		}
		return model;
	}
}
