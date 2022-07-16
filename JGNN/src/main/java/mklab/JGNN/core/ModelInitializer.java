package mklab.JGNN.core;

/**
 * This class defines an abstract interface of applying initializers to models.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class ModelInitializer {
	/**
	 * Applies the initializer to a given model's parameters.
	 * @param model The given model.
	 * @return The given model after parameters are initialized.
	 */
	public abstract Model apply(Model model);
	
}
