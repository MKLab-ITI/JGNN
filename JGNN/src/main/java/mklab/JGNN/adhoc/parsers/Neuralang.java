package mklab.JGNN.adhoc.parsers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.core.Tensor;

/**
 * Extends the base {@link ModelBuilder} with the full capabilities of the Neuralang 
 * scripting language.
 * 
 * @author Emmanouil Krasanakis
 * @see #parse(String)
 * @see #parse(Path)
 */
public class Neuralang extends ModelBuilder {
	public Neuralang() {
	}
	public Neuralang config(String name, double value) {
		super.config(name, value);
		return this;
	}
	/**
	 * Parses a Neuralang source code file.
	 * Reads a file like <code>Paths.get("models.nn")</code> 
	 * from disk with {@link Files#readAllLines(Path)}, and parses
	 * the loaded String.
	 * @param path The source code file.
	 * @return The Neuralang builder's instance.
	 * @see #parse(String)
	 */
	public Neuralang parse(Path path) {
		try {
			parse(String.join("\n", Files.readAllLines(path)));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return this;
	}
	
	/**
	 * Parses Neuralang source code by handling function declarations in addition to
	 * other expressions.
	 * @param text The source code to parse.
	 * @return The Neuralang builder's instance.
	 */
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
