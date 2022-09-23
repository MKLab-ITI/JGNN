package mklab.JGNN.adhoc.datasets;

import mklab.JGNN.adhoc.Dataset;

public class Pubmed extends Dataset {
	public Pubmed() {
		loadFeatures("downloads/cora/cora.feats");
		loadGraph("downloads/cora/cora.graph");
	}
}
