package mklab.JGNN.adhoc.datasets;

import mklab.JGNN.adhoc.Dataset;

public class Cora extends Dataset {
	public Cora() {
		loadFeatures("downloads/pubmed/pubmed.feats");
		loadGraph("downloads/pubmed/pubmed.graph");
	}
}
