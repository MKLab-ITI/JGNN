package mklab.JGNN.examples;

import java.util.HashMap;
import java.util.Map.Entry;

import mklab.JGNN.core.primitives.optimizers.Adam;
import mklab.JGNN.models.GCN;

public class LinkPrediction {

	public static void main(String[] args) throws Exception {
		Dataset dataset = new Datasets.FRIENDS();
		
		HashMap<String, Integer> nodeIds = new HashMap<String, Integer>();
		for(Entry<String, String> interaction : dataset.getInteractions()) {
			String u = interaction.getKey();
			String v = interaction.getValue();
			if(!nodeIds.containsKey(u))
				nodeIds.put(u, nodeIds.size());
			if(!nodeIds.containsKey(v))
				nodeIds.put(v, nodeIds.size());
			if(nodeIds.size()>20000)
				break;
		}
		GCN gcn = new GCN(nodeIds.size());
		for(Entry<String, String> interaction : dataset.getInteractions()) {
			if(nodeIds.get(interaction.getKey())==null || nodeIds.get(interaction.getValue())==null)
				continue;
			gcn.addEdge(nodeIds.get(interaction.getKey()), nodeIds.get(interaction.getValue()));
		}
		gcn.trainRelational(new Adam(0.01), 50);
	}
}
