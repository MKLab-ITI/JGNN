package mklab.JGNN.adhoc.datasets;

import mklab.JGNN.adhoc.Dataset;

public class Citeseer extends Dataset {
	public Citeseer() {
		loadFeatures("downloads/citeseer/citeseer.feats");
		loadGraph("downloads/citeseer/citeseer.graph");
	}
}
