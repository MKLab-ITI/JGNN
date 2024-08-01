package tutorial;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.datasets.Citeseer;
import mklab.JGNN.adhoc.datasets.Cora;
import mklab.JGNN.adhoc.parsers.FastBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.CategoricalCrossEntropy;
import mklab.JGNN.nn.optimizers.Adam;

/**
 * Demonstrates classification with an architecture defined through the scripting engine.
 * 
 * @author Emmanouil Krasanakis
 */
public class Quickstart {
	public static void main(String[] args) throws Exception {
		Dataset dataset = new Cora();
		Matrix adjacency = dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();
		Matrix nodeFeatures = dataset.features();
		Matrix nodeLabels = dataset.labels();
		Slice nodes = dataset.samples().getSlice().shuffle(100);
		long numClasses = nodeLabels.getCols();

		ModelBuilder modelBuilder = new FastBuilder(adjacency, nodeFeatures)
				.config("reg", 0.005)
				.config("hidden", 16)
				.config("classes", numClasses)
				.layer("h{l+1}=relu(h{l}@matrix(features, hidden, reg)+vector(hidden))")
				.layer("h{l+1}=h{l}@matrix(hidden, classes)+vector(classes)")
				.rememberAs("0")
				.constant("a", 0.9)
				.layerRepeat("h{l+1} = a*(dropout(A, 0.5)@h{l})+(1-a)*h{0}", 10)
				.classify();

		ModelTraining trainer = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(300)
				.setPatience(100)
				.setLoss(new CategoricalCrossEntropy())
				.setVerbose(true)
				.setValidationLoss(new CategoricalCrossEntropy());

		Model model = modelBuilder.getModel()
				.init(new XavierNormal())
				.train(trainer,
						nodes.samplesAsFeatures(), 
						nodeLabels, 
						nodes.range(0, 0.6), 
						nodes.range(0.6, 0.8));

		Matrix output = model.predict(nodes.samplesAsFeatures()).get(0).cast(Matrix.class);
		double acc = 0;
		for(Long node : nodes.range(0.8, 1)) {
			Matrix trueLabels = dataset.labels().accessRow(node).asRow();
			Tensor nodeOutput = output.accessRow(node).asRow();
			acc += nodeOutput.argmax()==trueLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/nodes.range(0.8, 1).size());
	}
}
