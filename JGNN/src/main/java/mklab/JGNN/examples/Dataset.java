package mklab.JGNN.examples;

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
import java.util.Map.Entry;

import mklab.JGNN.core.util.Sort;
import net.lingala.zip4j.ZipFile;

public class Dataset {
	private ArrayList<Entry<String, String>> interactions = new  ArrayList<Entry<String, String>>();
	
	protected Dataset() {
	}

	public Dataset(String path, String delimiter, int senderColumn, int receiverColumn, int timeColumn) {
		init(path, delimiter, senderColumn, receiverColumn, timeColumn);
	}
	
	protected static void downloadSource(String name, String url) {
		if(!(new File("datasets"+File.separator+name+".zip")).exists()) {
			try {
				(new File("datasets"+File.separator)).mkdirs();
				System.out.println("Downloading: "+url);
				ReadableByteChannel readableByteChannel = Channels.newChannel(new URL(url).openStream());
				FileOutputStream fileOutputStream = new FileOutputStream("datasets"+File.separator+name+".zip");
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
			System.out.println("Unzipping: "+"datasets"+File.separator+name+".zip");
			(new ZipFile(new File("datasets"+File.separator+name+".zip"))).extractAll("datasets"+File.separator);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	protected void init(String path, String delimiter, int senderColumn, int receiverColumn, int timeColumn) {
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
