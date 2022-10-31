package mklab.JGNN.adhoc;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.core.tensor.SparseTensor;

/**
 * This class provides the backbone with which to define datasets.
 * It also presents common operations.
 * @author Emmanouil Krasanakis
 * @see #samples()
 * @see #features()
 * @see #labels()
 * @see #graph()
 */
public class Dataset {
	private IdConverter nodes;
	private Matrix features;
	private IdConverter class2Ids;
	private Matrix labels;
	private Matrix graph;
	
	protected void downloadIfNotExists(String file, String url) {
		if(Files.exists(Paths.get(file)))
			return;
		System.out.println("First time requesting: "+url+"\nDownloading to: "+file);
		try {
			Files.createDirectories(Paths.get(file).getParent());
			ReadableByteChannel readableByteChannel = Channels.newChannel(new URL(url).openStream());
			try (FileOutputStream fileOutputStream = new FileOutputStream(file)) {
				fileOutputStream.getChannel()
				  .transferFrom(readableByteChannel, 0, Long.MAX_VALUE);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void loadFeatures(String file) {
		nodes = new IdConverter();
		ArrayList<Tensor> rows = new ArrayList<Tensor>();
		ArrayList<Integer> classes = new ArrayList<Integer>();
		class2Ids = new IdConverter();
		try(BufferedReader reader = new BufferedReader(new FileReader(file))){
			String line = reader.readLine();
			while (line != null) {
				String[] cols = line.split(",");
				if(cols.length<2)
					continue;
				nodes.getOrCreateId(cols[0]);
				Tensor features = new SparseTensor(cols.length-2);
				for(int col=0;col<cols.length-2;col++)
					features.put(col, Double.parseDouble(cols[col+1]));
				rows.add(features);
				classes.add((int)class2Ids.getOrCreateId(cols[cols.length-1]));
				line = reader.readLine();
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		features = new WrapRows(rows).toSparse();
		labels = new SparseMatrix(features.getRows(), class2Ids.size());
		for(int row=0;row<classes.size();row++)
			labels.put(row, classes.get(row), 1);
	}
	protected void loadGraph(String file) {
		graph = new SparseMatrix(nodes.size(), nodes.size());
		try(BufferedReader reader = new BufferedReader(new FileReader(file))){
			String line = reader.readLine();
			while (line != null) {
				String[] cols = line.split(",");
				if(cols.length<2)
					continue;
				long from = nodes.getId(cols[0]);
				long to = nodes.getId(cols[1]);
				graph.put(from, to, 1).put(to, from, 1);
				line = reader.readLine();
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Retrieves a converter that maps samples to long identifiers that match them to
	 * rows of {@link #features()}, {@link #labels()}, and {@link #graph()} matrices.
	 * For example, a list of all node ids can be obtained per
	 * <code>dataset.samples().getIds()</code>
	 * @return A {@link IdConverter}.
	 */
	public IdConverter samples() {
		return nodes;
	}
	/**
	 * Retrieves a converter that maps class names to label dimentions.
	 * For example, the prediction for one sample can be converted to its name 
	 * per <code>dataset.classes().get(prediction.argmax())</code>.
	 * @return An {@link IdConverter}.
	 */
	public IdConverter classes() {
		return class2Ids;
	}
	/**
	 * Retrieves the dataset's
	 * @return
	 */
	public Matrix features() {
		return features;
	}
	/**
	 * Retrieves the dataset's sample labels in one-hot encoding.
	 * @return A nodes x classes {@link Matrix}.
	 */
	public Matrix labels() {
		return labels;
	}
	/**
	 * Retrieves the dataset's graph.
	 * @return A {@link Matrix} or <code>null</code> if the dataset is feature-only.
	 */
	public Matrix graph() {
		return graph;
	}
}
