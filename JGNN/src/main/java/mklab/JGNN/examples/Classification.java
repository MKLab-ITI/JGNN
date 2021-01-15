package mklab.JGNN.examples;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.optimizers.Adam;
import mklab.JGNN.core.optimizers.Regularization;
import mklab.JGNN.core.tensor.AccessSubtensor;
import mklab.JGNN.core.tensor.RepeatTensor;
import mklab.JGNN.datasets.Dataset;
import mklab.JGNN.datasets.Datasets;
import mklab.JGNN.models.IdConverter;
import mklab.JGNN.models.relational.ClassificationGCN;

public class Classification {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodeIds = new IdConverter();
		HashMap<Integer, String> nodeLabels = new HashMap<Integer, String>();
		for(Entry<String, String> interaction : dataset.getInteractions()) {
			String u = interaction.getKey();
			String v = interaction.getValue();
			nodeLabels.put(nodeIds.getOrCreateId(u), dataset.getLabel(u));
			nodeLabels.put(nodeIds.getOrCreateId(v), dataset.getLabel(v));
		}
		Tensor allNodes = Tensor.fromRange(0, nodeLabels.size());
		List<Tensor> allLabels = nodeIds.toMultipleTensors(nodeLabels);
		long numTraining = (long)(allNodes.size()*0.9);
		Tensor trainingNodes = new AccessSubtensor(allNodes, 0, numTraining);
		List<Tensor> trainingLabels = new ArrayList<Tensor>(); 
		for(Tensor labels : allLabels)
			trainingLabels.add(new AccessSubtensor(labels, 0, numTraining));
		Tensor testNodes = new AccessSubtensor(allNodes, numTraining);
		List<Tensor> testLabels = new ArrayList<Tensor>(); 
		for(Tensor labels : allLabels)
			testLabels.add(new AccessSubtensor(labels, numTraining));
		
		ClassificationGCN gcn = new ClassificationGCN(nodeIds.oneHot(trainingLabels), 
											  new RepeatTensor(32, 15), 0);
		for(Entry<String, String> interaction : dataset.getInteractions()) 
			gcn.addEdge(nodeIds.getId(interaction.getKey()), nodeIds.getId(interaction.getValue()));
		gcn.trainClassification(new Regularization(new Adam(0.01), 5.E-4), 200, 
				trainingNodes, trainingLabels, testNodes, testLabels);
	}
}
