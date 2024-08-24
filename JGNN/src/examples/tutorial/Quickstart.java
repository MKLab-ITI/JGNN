package tutorial;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.adhoc.datasets.Cora;
import mklab.JGNN.adhoc.parsers.FastBuilder;
import mklab.JGNN.adhoc.parsers.Neuralang;
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
public class Quickstart {
	public static void main(String[] args) throws Exception {
		Dataset dataset = new Cora();
		dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();
		
		String architectures = """
			fn classify(nodes, h, epochs: !3000, patience: !100, lr: !0.01) {
				return softmax(h[nodes], dim: "row");
			}
			fn gcnlayer(A, h, hidden: 16, reg: 0.005) {
				return A@h@matrix(?, hidden, reg) + vector(hidden);
			}
			fn gcn(A, h, classes: extern) {
				h = gcnlayer(A, h);
				h = dropout(relu(h), 0.5);
				return gcnlayer(A, h, hidden: classes);
			}
		""";
		
		long numSamples = dataset.samples().getSlice().size();
		long numClasses = dataset.labels().getCols();
		ModelBuilder modelBuilder = new Neuralang()
				.parse(architectures)
				.constant("A", dataset.graph())
				.constant("h", dataset.features())
				.var("nodes")
				.config("classes", numClasses)
				.config("hidden", numClasses+2)
				.out("classify(nodes, gcn(A,h))")
				.autosize(new EmptyTensor(numSamples));
		System.out.println("Preferred learning rate: "+modelBuilder.getConfig("lr"));
		
		ModelTraining trainer = new ModelTraining()
				.configFrom(modelBuilder)
				.setLoss(new CategoricalCrossEntropy())
				.setValidationLoss(new VerboseLoss(new CategoricalCrossEntropy()));
		
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
