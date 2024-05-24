/*
 * UniqueNames.java
 *
 * Created on 13. listopad 2006, 17:13
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.utils;

/**
 * Simple class for giving unique ids. Each time the method getUniqueName is called,
 * a variable is incremented and the unique number is returned.
 * @author Ondra
 */
public class UniqueIDs {
    
    private static long unique = 1;
    
    /** Creates a new instance of UniqueNames */
    private UniqueIDs() {
    }
    
    /**
     * 
     * @return a unique long
     */
    public static long getUniqueName(){
        return unique++;
    }
}
