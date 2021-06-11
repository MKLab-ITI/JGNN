package mklab.JGNN.datasets;

import java.io.File;

public class Datasets {

	public static final class Cora extends Dataset {
		public Cora() {
			String folderName = "downloads/cora";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "cora", "https://data.deepai.org/Cora.zip");
			initStatic(folderName+"/Cora/edges.csv", ",", 0, 1);
			initNodeLabels(folderName+"/Cora/group-edges.csv", ",", 0, 1);
		}
	}
	
	
	public static final class CoraGraph extends Dataset {
		public CoraGraph() {
			String folderName = "downloads/coragraph";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "cora", "http://nrvis.com/download/data/labeled/cora.zip");
			initStatic(folderName+"/cora.edges", ",", 0, 1);
			initNodeLabels(folderName+"/cora.node_labels", ",", 0, 1);
		}
	}

	public static final class CiteSeer extends Dataset {
		public CiteSeer() {
			String folderName = "downloads/citeseer";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "citeseer", "http://nrvis.com/download/data/labeled/citeseer.zip");
			initStatic(folderName+"/citeseer.edges", ",", 0, 1);
			initNodeLabels(folderName+"/citeseer.node_labels", ",", 0, 1);
		}
	}

	public static final class PubMed extends Dataset {
		public PubMed() {
			String folderName = "downloads/pubmed";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "download", "http://nrvis.com/download/data/labeled/PubMed.zip");
			initStatic(folderName+"/PubMed.edges", ",", 0, 1);
			initNodeLabels(folderName+"/PubMed.node_labels", ",", 0, 1);
		}
	}
	
	public static final class MUTAG extends Dataset {
		public MUTAG() {
			String folderName = "downloads/mutag";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "download", "http://nrvis.com/download/data/labeled/Mutag.zip");
			initStatic(folderName+"/Mutag.edges", ",", 0, 1);
			initNodeLabels(folderName+"/Mutag.node_labels", ",", 0, 1);
		}
	}
	
	public static final class ENRON extends Dataset {
		public ENRON() {
			String folderName = "downloads/enron";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "download", "http://nrvis.com/download/data/ia/ia-enron-email-dynamic.zip");
			initTemporal(folderName+"/ia-enron-email-dynamic.edges", " ", 0, 1, 3);
		}
	}
	public static final class FRIENDS extends Dataset {
		public FRIENDS() {
			String folderName = "downloads/friends";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "download", "http://nrvis.com/download/data/dynamic/fb-wosn-friends.zip");
			initTemporal(folderName+"/fb-wosn-friends.edges", " ", 0, 1, 3);
		}
	}
	public static final class MESSAGES extends Dataset {
		public MESSAGES() {
			String folderName = "downloads/messages";
			if(!(new File(folderName)).exists())
				downloadSource(folderName, "download", "http://nrvis.com/download/data/dynamic/fb-messages.zip");
			initTemporal(folderName+"/fb-messages.edges", ",", 0, 1, 2);
		}
	}
	public static final class SMS extends Dataset {
		public SMS() {
			initTemporal("downloads/SMS.csv", ",", 1, 2, 0);
		}
	}
	public static final class CALLS extends Dataset {
		public CALLS() {
			initTemporal("downloads/calls.csv", ",", 1, 2, 0);
		}
	}
}
