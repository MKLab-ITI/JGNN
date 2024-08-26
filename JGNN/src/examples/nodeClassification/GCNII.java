package nodeClassification;

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
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.Accuracy;
import mklab.JGNN.nn.loss.CategoricalCrossEntropy;
import mklab.JGNN.nn.loss.report.VerboseLoss;
import mklab.JGNN.nn.optimizers.Adam;

public class GCNII {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Cora();
		dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();
		
		long numClasses = dataset.labels().getCols();
		ModelBuilder modelBuilder = new FastBuilder(dataset.graph(), dataset.features())
				.config("reg", 0.0005)
				.config("classes", numClasses)
				.config("hidden", numClasses)
				.constant("I", Matrix.eye(numClasses))
				.layer("h{l+1}=relu(h{l}@matrix(features, hidden, reg)+vector(hidden))")
				.rememberAs("0")
				.constant("a", 0.1)
				.futureConstants("b{l}", l->0.5/(l+1), 64)
				.layerRepeat("ch{l}=b{l}*matrix(hidden, hidden)+(1-b{l})*I; h{l+1}=relu((1-a)*(dropout(A, 0.5)@h{l})+a*h{0})@ch{l}", 64)
				.layer("h{l+1}=h{l}@matrix(hidden, classes)+vector(classes)")
				.classify()
				.assertBackwardValidity()
				.print();			;
		

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
		Matrix output = model.predict(Tensor.fromRange(0, nodes.size()).asColumn()).get(0).cast(Matrix.class);
		double acc = 0;
		for(Long node : nodes.range(0.8, 1)) {
			Matrix nodeLabels = dataset.labels().accessRow(node).asRow();
			Tensor nodeOutput = output.accessRow(node).asRow();
			acc += nodeOutput.argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/nodes.range(0.8, 1).size());
	}
}
