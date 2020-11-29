package mklab.JGNN.models;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.builders.GCNBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.Optimizer;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.AccessSubtensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Sort;

/**
 * Uses a {@Link GCNBuilder} to construct a Graph Convolutional Network
 * for unsupervised link prediction.
 * 
 * @author Emmanouil Krasanakis
 */
public class GCN extends Model {
	private Matrix W;
	private GCNBuilder builder;
	
	public GCN(List<Integer> layerDims) {
		W = new SparseMatrix(layerDims.get(0), layerDims.get(0));
		builder = new GCNBuilder(this, W, layerDims.get(1));
		for(int i=2;i<layerDims.size();i++)
			builder.aggregateAndTransform("tanh", layerDims.get(i));
		builder
			.similarity("distmult")
			.assertForwardValidity(Arrays.asList(3, 3))
			.assertBackwardValidity();
	}
	
	public void addEdge(int i, int j) {
		if(W.get(i, j)!=0)
			return;
		W.put(i, j, 1);
		//W.put(j, i, 1);
	}
	
	public double predict(int u, int v) {
		return predict(Arrays.asList(DenseTensor.fromDouble(u), DenseTensor.fromDouble(v)))
				.get(0) //first output
				.get(0);//first of the three tensor elements
	}
	
	protected void fillTrainingData(Tensor uList, Tensor vList, Tensor labels) {
		int pos = 0;
		for(Entry<Long, Long> edge : W.getNonZeroEntries()) {
			int u = (int)(long)edge.getKey();
			int v = (int)(long)edge.getValue();
			int retries = 1000;
			int neg1 = (int)(Math.random()*W.getRows());
			while(W.get(u, neg1)!=0 || neg1==u || neg1==v) {
				neg1 = (int)(Math.random()*W.getRows());
				retries -= 1;
				if(retries==0)
					throw new RuntimeException("Randomly sampling 100 nodes did not yield valid negative ones");
			}
			int neg2 = (int)(Math.random()*W.getRows());
			retries = 100;
			while(W.get(neg2, v)!=0 || neg2==u || neg2==v) {
				neg2 = (int)(Math.random()*W.getRows());
				retries -= 1;
				if(retries==0)
					throw new RuntimeException("Randomly sampling 100 nodes did not yield valid negative ones");
			}
			uList.put(pos, u);
			vList.put(pos, v);
			labels.put(pos, 1);
			pos += 1;

			uList.put(pos, u);
			vList.put(pos, neg1);
			pos += 1;
			
			uList.put(pos, neg2);
			vList.put(pos, v);
			pos += 1;
		}
	}

	public void trainRelational(Optimizer optimizer, int epochs) {
		trainRelational(optimizer, epochs, 0);
	}
	
	public void trainRelational(Optimizer optimizer, int epochs, double testSet) {
		W.setToLaplacian();
		boolean details = (testSet!=0);
		if(details) {
			builder.print();
			System.out.println("Number of nodes: "+W.getRows());
			System.out.println("Number of edges: "+W.getNumNonZeroElements());
		}
		for(int epoch=0;epoch<epochs;epoch++) {
			if(details)
				System.out.print("Epoch "+epoch);
			long numEdges = W.getNumNonZeroElements()*3;
			Tensor labels = new DenseTensor(numEdges);
			Tensor uList = new DenseTensor(numEdges);
			Tensor vList = new DenseTensor(numEdges);
			fillTrainingData(uList, vList, labels);
			// training
			long numTraining = (long)(numEdges * (1-testSet));
			List<Tensor> outputs = trainSample(optimizer,
												Arrays.asList(new AccessSubtensor(uList, 0, numTraining), 
															  new AccessSubtensor(vList, 0, numTraining)),
												Arrays.asList(new AccessSubtensor(labels, 0, numTraining)));
			if(details) {
				Tensor trainingPredictions = outputs.get(0);
				outputs = predict(Arrays.asList(new AccessSubtensor(uList, numTraining), 
												new AccessSubtensor(vList, numTraining)));
				Tensor testPredictions = outputs.get(0);
				System.out.println(
							" | training AUC "
							+auc(trainingPredictions, new AccessSubtensor(labels, 0, numTraining))
							+" | test AUC "
							+auc(testPredictions, new AccessSubtensor(labels, numTraining))
							);
			}
		}
	}

	public static double auc(Tensor predictions, Tensor labels) {
		if(labels.size()!=predictions.size())
			throw new RuntimeException("Predictions should have the same size as labels");
		long rank = 0;
		long positiveRankSum = 0;
		long n1 = 0;
		for(long idx : Sort.sortedIndexes(predictions.toArray())) {
			rank += 1;
			if(labels.get(idx)!=0) {
				//System.out.println(predictions.get(idx)+" "+labels.get(idx));
				positiveRankSum += rank;
				n1 += 1;
			}
		}
		positiveRankSum -= n1*(n1+1)/2;
		//System.out.println(predictions.size()+" "+n1+" "+positiveRankSum);
		long n2 = predictions.size()-n1;
		return positiveRankSum/(double)n1/n2;
	}
}
