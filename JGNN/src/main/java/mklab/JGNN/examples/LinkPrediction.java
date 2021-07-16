package mklab.JGNN.examples;

import java.util.HashMap;
import java.util.Map.Entry;

import mklab.JGNN.core.tensor.RepeatTensor;
import mklab.JGNN.data.IdConverter;
import mklab.JGNN.data.datasets.Dataset;
import mklab.JGNN.data.datasets.Datasets;
import mklab.JGNN.models.relational.RelationalGCN;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.Regularization;

public class LinkPrediction {

	public static void main(String[] args) throws Exception {
		/*Dataset dataset = new Datasets.CiteSeer();
		IdConverter nodeIds = new IdConverter();
		HashMap<Integer, String> nodeLabels = new HashMap<Integer, String>();
		for(Entry<String, String> interaction : dataset.getInteractions()) {
			String u = interaction.getKey();
			String v = interaction.getValue();
			nodeLabels.put(nodeIds.getOrCreateId(u), dataset.getLabel(u));
			nodeLabels.put(nodeIds.getOrCreateId(v), dataset.getLabel(v));
		}
		RelationalGCN gcn = new RelationalGCN(RelationalGCN.trueres_linear,
											  nodeIds.size(),//oneHot(nodeLabels), 
											  new RepeatTensor(16, 2));
		for(Entry<String, String> interaction : dataset.getInteractions()) 
			gcn.addEdge(nodeIds.getId(interaction.getKey()), nodeIds.getId(interaction.getValue()));
		gcn.trainRelational(new Regularization(new Adam(0.001), 5.E-4), 200, 0.2);*/
	}
}
