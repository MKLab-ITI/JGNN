package mklab.JGNN.adhoc.datasets;

import mklab.JGNN.adhoc.Dataset;

/**
 * Downloads and constructs the Cora node classification {@link Dataset}.
 * @author Emmanouil Krasanakis
 */
public class Cora extends Dataset {
	public Cora() {
		downloadIfNotExists("downloads/cora/cora.feats", 
				"https://github.com/maniospas/graph-data/raw/main/cora/cora.feats");
		downloadIfNotExists("downloads/cora/cora.graph", 
				"https://github.com/maniospas/graph-data/raw/main/cora/cora.graph");
		loadFeatures("downloads/cora/cora.feats");
		loadGraph("downloads/cora/cora.graph");
	}
}
