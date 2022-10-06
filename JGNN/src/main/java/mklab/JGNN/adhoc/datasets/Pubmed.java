package mklab.JGNN.adhoc.datasets;

import mklab.JGNN.adhoc.Dataset;

public class Pubmed extends Dataset {
	public Pubmed() {
		loadFeatures("downloads/pubmed/pubmed.csv");
		loadGraph("downloads/pubmed/pubmed.graph");
	}
}
