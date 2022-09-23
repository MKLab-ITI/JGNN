package nodeClassification;

import mklab.JGNN.adhoc.Dataset;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.datasets.Citeseer;
import mklab.JGNN.adhoc.parsers.GCNBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.Accuracy;
import mklab.JGNN.nn.loss.CategoricalCrossEntropy;
import mklab.JGNN.nn.optimizers.Adam;

public class HetGCN {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Citeseer();
		dataset.graph().setMainDiagonal(1).setToSymmetricNormalization();
		
		long numClasses = dataset.labels().getCols();
		ModelBuilder modelBuilder = new GCNBuilder(dataset.graph(), dataset.features())
				.config("reg", 0.005)
				.config("classes", numClasses)
				.config("hidden", numClasses)
				.constant("L", dataset.graph().negative().cast(Matrix.class).setMainDiagonal(1))
				.layer("h_low{l+1}=relu(dropout(A, 0.5)@(h{l}@matrix(features, hidden, reg))+vector(hidden));"
						+ "h_high{l+1}=relu(dropout(L, 0.5)@(h{l}@matrix(features, hidden, reg))+vector(hidden));"
						+ "h{l+1}=h_low{l+1}@matrix(hidden, hidden, reg)+h_high{l+1}@matrix(hidden, hidden, reg)")
				.layerRepeat("h_low{l+1}=relu(dropout(A, 0.5)@(h{l}@matrix(hidden, hidden, reg))+vector(hidden));"
						+ "h_high{l+1}=relu(dropout(L, 0.5)@(h{l}@matrix(hidden, hidden, reg))+vector(hidden));"
						+ "h{l+1}=h_low{l+1}@matrix(hidden, hidden, reg)+h_high{l+1}@matrix(hidden, hidden, reg)", 1)
				.classify()
				.assertBackwardValidity()
				.print();				;
		
		ModelTraining trainer = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(3000)
				.setPatience(30)
				.setLoss(new CategoricalCrossEntropy())
				.setValidationLoss(new Accuracy());
		
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
