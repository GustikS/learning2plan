/*
 * Term.java
 *
 * Created on 30. listopad 2006, 16:47
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.ilp.basic;

/**
 * Terms are variables, constants, function symbols and lists.
 * @author Ondra
 */
public interface Term {
    
    /**
     * 
     * @return string representation of the term
     */
    public String name();
    
}
