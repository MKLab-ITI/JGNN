package mklab.JGNN.data.datasets;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.util.Sort;
import mklab.JGNN.data.IdConverter;
import net.lingala.zip4j.ZipFile;

/**
 * Provides automatic downloading and importing of well-known public datasets. 
 * These can be used to test machine learning algorithms developed with JGNN.
 * If you use the provided datasets for publication, please follow the links
 * outputted by the library to the error console to find attribution requirements.
 * 
 * @author Emmanouil Krasanakis
 */
public class Dataset {
	private IdConverter nodeIds = new IdConverter();
	private HashMap<Long, String> nodeLabels = new HashMap<Long, String>();
	private HashMap<Long, ArrayList<String>> nodeFeatures = new HashMap<Long, ArrayList<String>>();
	private ArrayList<Entry<Long, Long>> interactions = new ArrayList<Entry<Long, Long>>();
	private int numFeatures = 0;
	
	protected Dataset() {
	}

	public Dataset(String path, String delimiter, int senderColumn, int receiverColumn, int timeColumn) {
		initTemporal(path, delimiter, senderColumn, receiverColumn, timeColumn);
	}
	
	public int getNumberOfFeatures() {
		if(numFeatures==0)
			throw new RuntimeException("Dataset "+toString()+" does not comprise node features");
		return numFeatures;
	}
	
	public IdConverter nodes() {
		if(nodeIds.size()==0)
			throw new RuntimeException("Dataset "+toString()+" does not comprise any nodes (this error should normally not appear)");
		return nodeIds;
	}

	public HashMap<Long, String> getLabels() {
		return nodeLabels;
	}
	
	public String getLabel(long node) {
		if(nodeLabels.isEmpty())
			throw new RuntimeException("Dataset "+toString()+" does not comprise node labels");
		return nodeLabels.get(node);
	}
	
	/**
	 * Retrieves the features of a single node.
	 * @param node The node.
	 * @return A list of String features.
	 */
	public ArrayList<String> getNodeFeatures(long node) {
		if(nodeFeatures.isEmpty())
			throw new RuntimeException("Dataset "+toString()+" does not comprise node features");
		return nodeFeatures.get(node);
	}
	
	/**
	 * Retrieves a list of maps from nodes to feature values. Each tensor
	 * corresponds to a different feature.
	 * @return A list of hashmaps from node identifiers to string values.
	 */
	public ArrayList<HashMap<Long, String>> getFeatures() {
		ArrayList<HashMap<Long, String>> ret = new ArrayList<HashMap<Long, String>>();
		for(int i=0;i<numFeatures;i++) {
			HashMap<Long, String> feature = new HashMap<Long, String>();
			for(Long node : nodeIds.getIds())
				if(getNodeFeatures(node)!=null)
					feature.put(node, getNodeFeatures(node).get(i));
			ret.add(feature);
		}
		return ret;
	}

	protected static void downloadSourceFile(String path, String url) {
		if(!(new File(path)).exists()) {
			try {
				(new File(path.substring(0, Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\')))+File.separator)).mkdirs();
				System.out.println("Downloading: "+url);
				ReadableByteChannel readableByteChannel = Channels.newChannel(new URL(url).openStream());
				FileOutputStream fileOutputStream = new FileOutputStream(path);
				FileChannel fileChannel = fileOutputStream.getChannel();
				fileOutputStream.getChannel()
				  .transferFrom(readableByteChannel, 0, Long.MAX_VALUE);
				fileChannel.close();
				fileOutputStream.close();
			}
			catch(Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	protected static void downloadSource(String path, String name, String url) {
		if(!(new File(path+File.separator+name+".zip")).exists()) {
			try {
				(new File(path+File.separator)).mkdirs();
				System.out.println("Downloading: "+url);
				ReadableByteChannel readableByteChannel = Channels.newChannel(new URL(url).openStream());
				FileOutputStream fileOutputStream = new FileOutputStream(path+File.separator+name+".zip");
				FileChannel fileChannel = fileOutputStream.getChannel();
				fileOutputStream.getChannel()
				  .transferFrom(readableByteChannel, 0, Long.MAX_VALUE);
				fileChannel.close();
				fileOutputStream.close();
			}
			catch(Exception e) {
				e.printStackTrace();
			}
		}
		try {
			System.out.println("Unzipping: "+path+File.separator+name+".zip");
			(new ZipFile(new File(path+File.separator+name+".zip"))).extractAll(path+File.separator);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void initStaticGraph(String path, String delimiter, int senderColumn, int receiverColumn) {
		try {
			BufferedReader edgeReader = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			while((line=edgeReader.readLine())!=null) {
				if(line.startsWith("%") || line.startsWith("#") || line.isEmpty())
					continue;
				String[] splt = line.split(delimiter);
				if(splt.length<2)
					continue;
				addInteraction(splt[senderColumn], splt[receiverColumn]);
			}
			edgeReader.close();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	

	protected void initNodeLabels(String path, String delimiter, int nodeColumn, int labelColumn) {
		try {
			BufferedReader labelReader = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			while((line=labelReader.readLine())!=null) {
				if(line.startsWith("%") || line.startsWith("#") || line.isEmpty())
					continue;
				String[] splt = line.split(delimiter);
				if(splt.length<2)
					continue;
				if(labelColumn<0)
					labelColumn = splt.length + labelColumn;
				String nodeName = nodeColumn==-1?"Node "+nodeIds.size():splt[nodeColumn];
				setLabel(nodeName, splt[labelColumn]);
				ArrayList<String> features = new ArrayList<String>();
				for(int i=0;i<splt.length;i++)
					if(i!=nodeColumn && i!=labelColumn)
						features.add(splt[i]);
				setFeatures(nodeName, features);
			}
			labelReader.close();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void initStaticLabeled(String path, String labelPath, String delimiter, int senderColumn, int receiverColumn, int labelColumn) {
		try {
			BufferedReader edgeReader = new BufferedReader(new FileReader(new File(path)));
			//BufferedReader labelReader = new BufferedReader(new FileReader(new File(labelPath)));
			String line = null;
			while((line=edgeReader.readLine())!=null) {
				//String labelLine = labelReader.readLine();
				if(line.startsWith("%") || line.startsWith("#") || line.isEmpty())
					continue;
				String[] splt = line.split(delimiter);
				if(splt.length<2)
					continue;
				//String[] spltLabel = line.split(labelLine);
				
				addInteraction(splt[senderColumn], splt[receiverColumn]);
			}
			//labelReader.close();
			edgeReader.close();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void initTemporal(String path, String delimiter, int senderColumn, int receiverColumn, int timeColumn) {
		try {
			ArrayList<String> interactions = new ArrayList<String>();
			BufferedReader edgeReader = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			ArrayList<Double> timestamps = new ArrayList<Double>();
			while((line=edgeReader.readLine())!=null) {
				if(line.startsWith("%") || line.startsWith("#") || line.isEmpty())
					continue;
				String[] splt = line.split(delimiter);
				if(splt.length<3)
					continue;
				String time = splt[timeColumn];
				if(time.equals("0"))
					continue;
				interactions.add(splt[0]+" "+splt[1]+" "+time);
				timestamps.add(Double.parseDouble(time));
			}
			edgeReader.close();
			for(int i : Sort.sortedIndexes(timestamps)) {
				String[] splt = interactions.get(i).split(" ");
				String u = splt[0];
				String v = splt[1];
				addInteraction(u, v);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void addInteraction(String u, String v) {
		this.interactions.add(new AbstractMap.SimpleEntry<Long, Long>(nodeIds.getOrCreateId(u), nodeIds.getOrCreateId(v)));
		
	}
	protected void setLabel(String u, String label) {
		this.nodeLabels.put(nodeIds.getOrCreateId(u), label);
	}
	protected void setFeatures(String u, ArrayList<String> features) {
		if(numFeatures==0)
			numFeatures = features.size();
		else if(numFeatures!=features.size())
			throw new RuntimeException("Entry with "+features.size()+" features instread of "+numFeatures+" found in previous entry");
		this.nodeFeatures.put(nodeIds.getOrCreateId(u), features);
	}
	
	public ArrayList<Entry<Long, Long>> getInteractions() {
		if(interactions.isEmpty())
			throw new RuntimeException("Dataset "+toString()+" does not comprise interactions");
		return interactions;
	}
}
