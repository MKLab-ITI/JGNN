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

import mklab.JGNN.core.util.Sort;
import net.lingala.zip4j.ZipFile;

public class Dataset {
	private ArrayList<Entry<String, String>> interactions = new  ArrayList<Entry<String, String>>();
	private HashMap<Object, String> labels = new HashMap<Object, String>();
	
	protected Dataset() {
	}

	public Dataset(String path, String delimiter, int senderColumn, int receiverColumn, int timeColumn) {
		initTemporal(path, delimiter, senderColumn, receiverColumn, timeColumn);
	}
	
	public String getLabel(Object obj) {
		return labels.get(obj);
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
	
	protected void initStatic(String path, String delimiter, int senderColumn, int receiverColumn) {
		try {
			BufferedReader edgeReader = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			while((line=edgeReader.readLine())!=null) {
				if(line.startsWith("%") || line.startsWith("#") || line.isEmpty())
					continue;
				String[] splt = line.split(delimiter);
				if(splt.length<2)
					continue;

				this.interactions.add(new AbstractMap.SimpleEntry<String, String>(splt[senderColumn], splt[receiverColumn]));
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
				this.labels.put(splt[nodeColumn], splt[labelColumn]);
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
				
				Entry<String, String> interaction = new AbstractMap.SimpleEntry<String, String>(splt[senderColumn], splt[receiverColumn]);
				this.interactions.add(interaction);
				this.labels.put(interaction, "0");//spltLabel[labelColumn]);
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
				this.interactions.add(new AbstractMap.SimpleEntry<String, String>(u, v));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public ArrayList<Entry<String, String>> getInteractions() {
		return interactions;
	}
	
}
