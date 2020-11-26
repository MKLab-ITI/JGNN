package mklab.JGNN.examples;

import java.io.File;

public class Datasets {
	public static final class ENRON extends Dataset {
		public ENRON() {
			if(!(new File("datasets/ia-enron-email-dynamic.edges")).exists())
				downloadSource("ia-enron-email-dynamic", "http://nrvis.com/download/data/ia/ia-enron-email-dynamic.zip");
			init("datasets/ia-enron-email-dynamic.edges", " ", 0, 1, 3);
		}
	}
	public static final class FRIENDS extends Dataset {
		public FRIENDS() {
			if(!(new File("datasets/fb-wosn-friends.edges")).exists())
				downloadSource("fb-wosn-friends", "http://nrvis.com/download/data/dynamic/fb-wosn-friends.zip");
			init("datasets/fb-wosn-friends.edges", " ", 0, 1, 3);
		}
	}
	public static final class MESSAGES extends Dataset {
		public MESSAGES() {
			if(!(new File("datasets/fb-messages.edges")).exists())
				downloadSource("fb-messages", "http://nrvis.com/download/data/dynamic/fb-messages.zip");
			init("datasets/fb-messages.edges", ",", 0, 1, 2);
		}
	}
	public static final class SMS extends Dataset {
		public SMS() {
			init("datasets/SMS.csv", ",", 1, 2, 0);
		}
	}
	public static final class CALLS extends Dataset {
		public CALLS() {
			init("datasets/calls.csv", ",", 1, 2, 0);
		}
	}
}
