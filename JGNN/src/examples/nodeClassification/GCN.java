package nodeClassification;

import java.nio.file.Files;
import java.nio.file.Paths;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.adhoc.datasets.Cora;
import mklab.JGNN.adhoc.parsers.FastBuilder;
import mklab.JGNN.adhoc.train.SampleClassification;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.empy.EmptyTensor;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.Accuracy;
import mklab.JGNN.nn.loss.CategoricalCrossEntropy;
import mklab.JGNN.nn.loss.report.VerboseLoss;
import mklab.JGNN.nn.optimizers.Adam;

/**
 * Demonstrates classification with the GCN architecture.
 * 
 * @author Emmanouil Krasanakis
 */
public class GCN {
	public static void main(String[] args) throws Exception {
		Dataset dataset = new Cora();
		dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();

		long numSamples = dataset.samples().getSlice().size();
		long numClasses = dataset.labels().getCols();
		ModelBuilder modelBuilder = new FastBuilder(dataset.graph(), dataset.features())
			.config("reg", 0.005)
			.config("classes", numClasses)
			.config("hidden", 64)
			.function("gcnlayer", "(A,h){Adrop = dropout(A, 0.5); return Adrop@(h@matrix(?, hidden, reg))+vector(?);}")
			.layer("h{l+1}=relu(gcnlayer(A, h{l}))")
			.config("hidden", "classes")  // reassigns the output gcnlayer's "hidden" to be the number of "classes"
			.layer("h{l+1}=gcnlayer(A, h{l})")
			.classify()
			.autosize(new EmptyTensor(numSamples));
		
		Slice nodes = dataset.samples().getSlice().shuffle(100);
		ModelTraining trainer = new SampleClassification()
				// set data
				.setFeatures(nodes.samplesAsFeatures())
				.setOutputs(dataset.labels())
				.setTrainingSamples(nodes.range(0, 0.6))
				.setValidationSamples(nodes.range(0.6, 0.8))
				// configure how training is conducted
				.setOptimizer(new Adam(0.01))
				.setEpochs(300)
				.setPatience(100)
				.setLoss(new CategoricalCrossEntropy())
				.setValidationLoss(new VerboseLoss(new CategoricalCrossEntropy(), new Accuracy()));
		
		long tic = System.currentTimeMillis();
		Model model = modelBuilder.getModel()
				.init(new XavierNormal())
				.train(trainer);
		
		System.out.println("Training time "+(System.currentTimeMillis()-tic)/1000.);
		//modelBuilder.save(Paths.get("gcn_cora.jgnn"));
		
		Model loadedModel = model;//ModelBuilder.load(Paths.get("gcn_cora.jgnn")).getModel();
		Matrix output = loadedModel.predict(Tensor.fromRange(0, nodes.size()).asColumn()).get(0).cast(Matrix.class);
		double acc = 0;
		for(Long node : nodes.range(0.8, 1)) {
			Matrix nodeLabels = dataset.labels().accessRow(node).asRow();
			Tensor nodeOutput = output.accessRow(node).asRow();
			acc += nodeOutput.argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/nodes.range(0.8, 1).size());
	}
}
