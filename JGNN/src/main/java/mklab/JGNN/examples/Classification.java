package mklab.JGNN.examples;

import java.util.Arrays;
import java.util.HashMap;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.nn.optimizers.BatchOptimizer;
import mklab.JGNN.nn.optimizers.GradientDescent;
import mklab.JGNN.nn.optimizers.Regularization;

public class Classification {

	public static void main(String[] args) {
		Dataset dataset = new Datasets.Lymphography();
		Matrix labels = dataset.nodes().oneHot(dataset.labels()).asTransposed();
		Matrix features = dataset.nodes().oneHot(dataset.features()).asTransposed();
		
		System.out.println("Nodes\t: "+dataset.nodes().size());
		System.out.println("Labels\t: "+labels.describe());
		System.out.println("Features: "+features.describe());
		
		long numFeatures = features.getRows();
		long numClasses = labels.getRows();
		int dims = 2;
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x")
				.param("w1", new DenseMatrix(numClasses, numFeatures).setToRandom().selfAdd(-0.5).selfMultiply(Math.sqrt(1./dims)))
				.operation("yhat = sigmoid(w1@x)")
				.out("yhat")
				.print();
		
		BatchOptimizer optimizer = new BatchOptimizer(new Regularization(new GradientDescent(0.1), 0.001));
		for(int epoch=0;epoch<150;epoch++) {
			System.out.print("Epoch "+epoch);

			Tensor errors = 
					modelBuilder.getModel().trainSampleDifference(optimizer, 
					Arrays.asList(features), 
					Arrays.asList(labels))
					.get(0)
					.subtract(labels);
			
			double acc = 0;
			for(Integer node : dataset.nodes().getIds()) {
				Matrix nodeFeatures = features.getCol(node).asColumn();
				Matrix nodeLabels = labels.getCol(node).asColumn();
				Tensor output = modelBuilder.getModel().predict(nodeFeatures).get(0);
				acc += (output.argmax()==nodeLabels.argmax()?1:0);
			}
			/*int prev = 0;
			int i = 0;
			for(Integer node : dataset.nodes().getIds()) {
				if(i-prev>dataset.nodes().size()/40) {
					System.out.print("=");
					prev = i;
				}
				i++;
				Matrix nodeFeatures = features.getRow(node).asColumn();
				Matrix nodeLabels = labels.getRow(node).asColumn();
				Tensor output = modelBuilder.getModel().trainSampleDifference(optimizer, 
						Arrays.asList(nodeFeatures), 
						Arrays.asList(nodeLabels)).get(0);
				Tensor error = output.subtract(nodeLabels);
				errors.selfAdd(error.norm());
				acc.selfAdd(output.argmax()==nodeLabels.argmax()?1:0);
				//System.out.println(output.toString()+" "+nodeLabels.toString());
				//System.out.println(output.argmax()+" "+nodeLabels.argmax());
			}*/
			optimizer.updateAll();
			System.out.print("\t error "+errors.abs().sum()/dataset.nodes().size());
			System.out.println("\t accuracy "+acc/dataset.nodes().size());
		}
	}

}
