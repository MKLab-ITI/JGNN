package mklab.JGNN.adhoc.datasets;

import mklab.JGNN.adhoc.Dataset;

public class Cora extends Dataset {
	public Cora() {
		loadFeatures("downloads/cora/cora.feats");
		loadGraph("downloads/cora/cora.graph");
	}
}
