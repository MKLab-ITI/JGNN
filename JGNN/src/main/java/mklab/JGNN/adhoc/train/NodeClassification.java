package mklab.JGNN.adhoc.train;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Memory;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.ThreadPool;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.inputs.Parameter;

/**
 * Extends the {@link ModelTraining} class with a method to explicitly train a
 * model from feature and label matrices.
 * 
 * @author Emmanouil Krasanakis
 * @see #train(Model, Matrix, Matrix, Slice, Slice)
 */
public class NodeClassification extends ModelTraining {

	/**
	 * Trains a {@link Model} instance based on current settings. The graph is
	 * assumed to be known and declared as an architecture constant.
	 * 
	 * @param model           The model instance to train.
	 * @param features        A matrix whose columns correspond to sample features.
	 * @param labels          A matrix whose columns correspond to sample (one hot)
	 *                        labels.
	 * @param trainingSamples Which columns to select for training.
	 * @return The trained <code>model</code> (the same instance as the first
	 *         argument).
	 */
	public Model train(Model model, Matrix features, Matrix labels, Slice trainingSamples, Slice validationSamples) {
		// ACTUΑL TRAINING
		double minLoss = Double.POSITIVE_INFINITY;
		HashMap<Parameter, Tensor> minLossParameters = new HashMap<Parameter, Tensor>();
		int currentPatience = patience;
		for (int epoch = 0; epoch < epochs; epoch++) {
			// long tic = System.currentTimeMillis();
			if (!stochasticGradientDescent)
				trainingSamples.shuffle(epoch);
			double[] batchLosses = new double[numBatches];
			for (int batch = 0; batch < numBatches; batch++) {
				if (stochasticGradientDescent)
					trainingSamples.shuffle(epoch);
				int start = (trainingSamples.size() / numBatches) * batch;
				int end = Math.min(trainingSamples.size(), start + (trainingSamples.size() / numBatches));
				int batchId = batch;
				Matrix trainFeatures = new WrapRows(features.accessRows(trainingSamples.range(start, end)))
						.setDimensionName(features.getRowName(), features.getColName());
				Matrix trainLabels = new WrapRows(labels.accessRows(trainingSamples.range(start, end)));
				// .setDimensionName(labels.getRowName(), labels.getColName());
				// System.out.println(System.currentTimeMillis()-tic);
				Runnable batchCode = new Runnable() {
					@Override
					public void run() {
						List<Tensor> outputs;
						outputs = model.train(loss, optimizer, Arrays.asList(trainFeatures),
								Arrays.asList(trainLabels));
						if (stochasticGradientDescent)
							optimizer.updateAll();
						if (validationSamples == null)
							batchLosses[batchId] = loss.evaluate(outputs.get(0), trainLabels);
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
			double totalLoss = 0;
			if (validationSamples == null)
				for (double batchLoss : batchLosses)
					totalLoss += batchLoss / numBatches;
			else {
				Memory.scope().enter();
				Matrix validationFeatures = new WrapRows(features.accessRows(validationSamples));
				Matrix validationLabels = new WrapRows(labels.accessRows(validationSamples));
				List<Tensor> outputs = model.predict(Arrays.asList(validationFeatures));
				totalLoss = (validationLoss != null ? validationLoss : loss).evaluate(outputs.get(0), validationLabels);// outputs.get(0).multiply(-1).cast(Matrix.class).selfAdd(validationLabels).selfAbs().norm();
				Memory.scope().exit();
				// for(long i=0;i<validationLabels.getRows();i++)
				// totalLoss
				// -=(outputs.get(0).cast(Matrix.class).accessRow(i).argmax()==validationLabels.accessRow(i).argmax())?1:0;
				// totalLoss /= validationLabels.getRows();
			}
			if (totalLoss < minLoss) {
				currentPatience = patience;
				minLoss = totalLoss;
				for (Parameter parameter : model.getParameters())
					minLossParameters.put(parameter, parameter.get().copy());
			}
			if (verbose)
				System.out.println("Epoch " + epoch + " with loss " + totalLoss);
			currentPatience -= 1;
			if (currentPatience == 0)
				break;
		}
		for (Parameter parameter : model.getParameters())
			parameter.set(minLossParameters.get(parameter));
		return model;
	}

}
