package mklab.JGNN.adhoc.parsers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.core.Tensor;

public class Neuralang extends ModelBuilder {
	public Neuralang() {
	}
	public Neuralang config(String name, double value) {
		super.config(name, value);
		return this;
	}
	public Neuralang parse(Path path) {
		try {
			parse(String.join("\n", Files.readAllLines(path)));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return this;
	}
	
	public Neuralang parse(String text) {
		int depth = 0;
		String progress = "";
		for(int i=0;i<text.length();i++) {
			char c =  text.charAt(i);
			if(c=='{')
				depth += 1;
			if(c=='}')
				depth -= 1;
			if((c==';' || c=='}') && depth==0) {
				progress += c;
				progress = progress.trim();
				if(progress.startsWith("fn ")) {
					function(progress.substring(3, progress.indexOf("(")).trim(), 
							      progress.substring(progress.indexOf("(")));
				}
				else if(!progress.isEmpty())
					operation(progress);
				progress = "";
				continue;
			}
			progress += c;
		}
		progress = progress.trim();
		if(progress.startsWith("fn ")) {
			function(progress.substring(3, progress.indexOf("(")).trim(), 
					      progress.substring(progress.indexOf("(")));
		}
		else if(!progress.isEmpty())
			operation(progress);
		return this;
	}
	public Neuralang constant(String name, Tensor value) {
		super.constant(name, value);
		return this;
	}
	public Neuralang constant(String name, double value) {
		super.constant(name, value);
		return this;
	}
	public Neuralang var(String var) {
		super.var(var);
		return this;
	}
	

}
