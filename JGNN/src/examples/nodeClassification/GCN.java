package nodeClassification;

import java.nio.file.Files;
import java.nio.file.Paths;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.datasets.Cora;
import mklab.JGNN.adhoc.parsers.FastBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.empy.EmptyTensor;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.CategoricalCrossEntropy;
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

		long numClasses = dataset.labels().getCols();
		ModelBuilder modelBuilder = new FastBuilder(dataset.graph(), dataset.features())
				.config("reg", 0.005)
				.config("classes", numClasses)
				.config("hidden", numClasses)
				.function("gcnlayer", "(A,h){z=dropout(A, 0.5)@(h@matrix(?, hidden, reg))+vector(?);return z}")
				.layer("h{l+1}=relu(gcnlayer(A, h{l}))")
				.config("hidden", "classes")
				.layer("h{l+1}=gcnlayer(A, h{l})")
				.classify()
				.autosize(new EmptyTensor(dataset.samples().getSlice().size()));
		
		ModelTraining trainer = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(10)
				.setPatience(100)
				.setVerbose(true)
				.setLoss(new CategoricalCrossEntropy())
				.setValidationLoss(new CategoricalCrossEntropy());
		
		long tic = System.currentTimeMillis();
		Slice nodes = dataset.samples().getSlice().shuffle(100);
		Model model = modelBuilder.getModel()
				.init(new XavierNormal())
				.train(trainer,
						Tensor.fromRange(nodes.size()).asColumn(), 
						dataset.labels(), nodes.range(0, 0.6), nodes.range(0.6, 0.8));
		
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
