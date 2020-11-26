package mklab.JGNN.models;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.operations.Model;
import mklab.JGNN.core.operations.ModelBuilder;
import mklab.JGNN.core.primitives.Matrix;
import mklab.JGNN.core.primitives.Optimizer;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.primitives.matrix.DenseMatrix;
import mklab.JGNN.core.primitives.matrix.SparseMatrix;
import mklab.JGNN.core.primitives.tensor.DenseTensor;
import mklab.JGNN.core.util.Sort;

public class GCN extends Model {
	private Matrix W;
	private ModelBuilder builder;
	
	public GCN(int numNodes) {
		W = new SparseMatrix(numNodes, numNodes);
		int dims = 32;
		builder = new ModelBuilder(this)
				.var("u")
				.var("v")
				.constant("W", W)
				.param("H0", new DenseMatrix(numNodes, dims).setToRandom().setToNormalized())
				.param("DistMult", new DenseTensor(dims).setToRandom().setToNormalized())
				//.param("B1", new DenseTensor(dims))
				.param("W1", new DenseMatrix(dims, dims).setToRandom().setToNormalized())
				.operation("H1 = W * H0 * W1")
				.operation("sim = sigmoid( sum(H1[u].H1[v].DistMult) )")
				.out("sim")
				//.assertForwardValidity(Arrays.asList(3, 3))
				.assertBackwardValidity();
	}
	
	public void addEdge(int i, int j) {
		if(W.get(i, j)!=0)
			return;
		//W.put(i, i, 1);
		//W.put(j, j, 1);
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
		trainRelational(optimizer, epochs);
	}
	
	public void trainRelational(Optimizer optimizer, int epochs, double testSet) {
		boolean details = (testSet==0);
		if(details) {
			builder.print();
			System.out.println("Number of nodes: "+W.getRows());
			System.out.println("Number of edges: "+W.getNumNonZeroElements());
		}
		optimizer.reset();
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
												Arrays.asList(uList.subtensor(0, numTraining), 
															  vList.subtensor(0, numTraining)),
												Arrays.asList(labels.subtensor(0, numTraining)));
			if(details) {
				Tensor trainingPredictions = outputs.get(0);
				outputs = predict(Arrays.asList(uList.subtensor(numTraining, numEdges), 
						  						vList.subtensor(numTraining, numEdges)));
				Tensor testPredictions = outputs.get(0);
				System.out.println(
							" | training AUC "
							+auc(trainingPredictions, labels.subtensor(0, numTraining))
							+" | test AUC "
							+auc(testPredictions, labels.subtensor(numTraining, numEdges))
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
