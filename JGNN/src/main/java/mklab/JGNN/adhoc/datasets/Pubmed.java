package mklab.JGNN.adhoc.datasets;

import mklab.JGNN.adhoc.Dataset;

/**
 * Downloads and constructs the Pubmed node classification {@link Dataset}.
 * 
 * @author Emmanouil Krasanakis
 */
public class Pubmed extends Dataset {
	public Pubmed() {
		downloadIfNotExists("downloads/pubmed/pubmed.feats",
				"https://github.com/maniospas/graph-data/raw/main/pubmed/pubmed.feats");
		downloadIfNotExists("downloads/pubmed/pubmed.graph",
				"https://github.com/maniospas/graph-data/raw/main/pubmed/pubmed.graph");
		loadFeatures("downloads/pubmed/pubmed.feats");
		loadGraph("downloads/pubmed/pubmed.graph");
	}
}
