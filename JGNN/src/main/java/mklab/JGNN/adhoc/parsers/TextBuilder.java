package mklab.JGNN.adhoc.parsers;

import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.core.Tensor;

public class TextBuilder extends ModelBuilder {
	public TextBuilder() {
	}
	public TextBuilder config(String name, double value) {
		super.config(name, value);
		return this;
	}
	public TextBuilder parse(String text) {
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
	public TextBuilder constant(String name, Tensor value) {
		super.constant(name, value);
		return this;
	}
	public TextBuilder constant(String name, double value) {
		super.constant(name, value);
		return this;
	}
	public TextBuilder var(String var) {
		super.var(var);
		return this;
	}
	

}
