package nodeClassification;

import java.util.Map.Entry;

import mklab.JGNN.adhoc.builders.GCNBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelBuilder;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.nn.NNOperation;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.loss.Accuracy;
import mklab.JGNN.core.loss.CategoricalCrossEntropy;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.optimizers.Adam;

public class APPNP {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		Matrix labels = dataset.nodes().oneHot(dataset.getLabels());
		Matrix features = dataset.nodes().oneHotFromBinary(dataset.getFeatures());
	
		Matrix adjacency = new SparseMatrix(dataset.nodes().size(), dataset.nodes().size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) 
			adjacency
				.put(interaction.getKey(), interaction.getValue(), 1)
				.put(interaction.getValue(), interaction.getKey(), 1);
		adjacency.setMainDiagonal(1).setToSymmetricNormalization();
		
		long numClasses = labels.getCols();
		ModelBuilder modelBuilder = new GCNBuilder(adjacency, features)
				.config("reg", 0.005)
				.config("classes", numClasses)
				.layer("h{l+1}=relu(h{l}@matrix(features, 32, reg)+vector(32))")
				.layer("h{l+1}=h{l}@matrix(32, classes)+vector(classes)")
				.rememberAs("0")
				.constant("a", 0.9)
				.layerRepeat("h{l+1} = a*(dropout(A, 0.5)@h{l})+(1-a)*h{0}", 10)
				.classify();				;
		
		ModelTraining trainer = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(300)
				.setPatience(100)
				.setLoss(new CategoricalCrossEntropy())
				.setValidationLoss(new Accuracy());
		
		Slice nodes = dataset.nodes().getIds().shuffle(100);
		Model model = modelBuilder.getModel()
				.init(new XavierNormal())
				.train(trainer,
						Tensor.fromRange(0, nodes.size()).asColumn(), 
						labels, nodes.range(0, 0.2), nodes.range(0.2, 0.4));
		
		
		Matrix output = model.predict(Tensor.fromRange(0, nodes.size()).asColumn()).get(0).cast(Matrix.class);
		double acc = 0;
		for(Long node : nodes.range(0.4, 1)) {
			Matrix nodeLabels = labels.accessRow(node).asRow();
			Tensor nodeOutput = output.accessRow(node).asRow();
			acc += nodeOutput.argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/nodes.range(0.4, 1).size());
	}
}
