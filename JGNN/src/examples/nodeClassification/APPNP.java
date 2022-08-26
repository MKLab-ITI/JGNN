package nodeClassification;

import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.ModelBuilder;
import mklab.JGNN.nn.ModelTraining;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.loss.Accuracy;
import mklab.JGNN.core.loss.BinaryCrossEntropy;
import mklab.JGNN.core.loss.CategoricalCrossEntropy;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.initializers.XavierUniform;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.GradientDescent;

public class APPNP {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();		
		Slice nodes = dataset.nodes().getIds().shuffle();
		
		Matrix labels = dataset.nodes()
				.oneHot(dataset.getLabels())
				.setDimensionName("nodes", "labels");
		Matrix features = dataset.nodes().oneHotFromBinary(dataset.getFeatures());
		
		Matrix adjacency = new SparseMatrix(dataset.nodes().size(), dataset.nodes().size());
		for(Entry<Long, Long> interaction : dataset.getInteractions()) {
			adjacency.put(interaction.getKey(), interaction.getValue(), 1);
			adjacency.put(interaction.getValue(), interaction.getKey(), 1);
		}
		adjacency.setMainDiagonal(1).setToLaplacian().setDimensionName("nodes", "nodes");
		
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		for(Tensor row : features.accessColumns()) 
			row.setToProbability();
		
		
		long numFeatures = features.getCols();
		long numClasses = labels.getCols();
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("nodes")
				.constant("A", adjacency)
				.constant("x", features)
				.constant("a", 0.9)
				.config("hidden", 64)
				.config("classes", numClasses)
				.config("feats", numFeatures)
				.config("reg", 0.0005)
				.operation("h = relu(x@mat(feats, hidden) + vec(hidden))")
				.operation("yhat = h@mat(hidden, classes) + vec(classes)")
				/*.operation("prop1 = a*A@yhat+(1-a)*yhat")
				.operation("prop2 = a*A@prop1+(1-a)*yhat")
				.operation("prop3 = a*A@prop2+(1-a)*yhat")
				.operation("prop4 = a*A@prop3+(1-a)*yhat")
				.operation("prop5 = a*A@prop4+(1-a)*yhat")
				.operation("prop6 = a*A@prop5+(1-a)*yhat")
				.operation("prop7 = a*A@prop6+(1-a)*yhat")
				.operation("prop8 = a*A@prop7+(1-a)*yhat")
				.operation("prop9 = a*A@prop8+(1-a)*yhat")*/
				.operation("yfinal = softmax(yhat, row)")
				.operation("node_yhat = yfinal[nodes]")
				.out("node_yhat");
		
		//System.out.println(modelBuilder.getExecutionGraphDot());
		
		new XavierUniform().apply(modelBuilder.getModel());

		Matrix nodeIdMatrix = nodes.asTensor().asColumn();
		Model model = new ModelTraining()
				.setOptimizer(new Adam(0.01))
				.setEpochs(3000)
				.setPatience(100)
				.setLoss(new CategoricalCrossEntropy())
				.setValidationLoss(new Accuracy())
				.train(modelBuilder.getModel(), nodeIdMatrix, labels, 
						nodes.range(0, 0.2), nodes.range(0.2, 0.4));
		
		nodeIdMatrix = nodes.asTensor().asColumn();
		double acc = 0;
		Matrix output = model.predict(nodeIdMatrix).get(0).cast(Matrix.class);
		for(Long i : Tensor.fromRange(0, nodes.range(0.4, 1).size())) {
			Matrix nodeLabels = labels.accessRow((long)nodeIdMatrix.get(i)).asRow();
			acc += output.accessRow(i).argmax()==nodeLabels.argmax()?1:0;
		}
		System.out.println("Acc\t "+acc/nodes.range(0.4, 1).size());
	}
}
