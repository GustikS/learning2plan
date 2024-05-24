/*
 * Variable.java
 *
 * Created on 30. listopad 2006, 16:35
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.ilp.basic;

import java.util.WeakHashMap;

/**
 * Class for representing first-order-logic variables.
 * 
 * @author Ondra
 */
public class Variable implements Term {
    
    private String name;
    
    private int hashCode = Integer.MIN_VALUE;
    
    private static WeakHashMap<String,Variable> cache = new WeakHashMap<String,Variable>();
    
    /** Creates a new instance of Constant */
    private Variable(String name) {
        this.name = name.trim().intern();
    }
    
    /**
     * Creates a new variable - it uses caching so that variables of the same name would be
     * represented by one object.
     * @param name name of the variable to be constructed
     * @return constructed variable (either cached or created as new)
     */
    public static Variable construct(String name){
        Variable retVal = cache.get(name);
        if (retVal == null){
            retVal = new Variable(name);
            cache.put(name, retVal);
        }
        return retVal;
    }
    
    public String name(){
        return name;
    }
    
    @Override
    public String toString(){
        return name;
    }

    @Override
    public boolean equals(Object o){
        if (o instanceof Variable){
            return o == this || ((Variable)o).name.equals(this.name);
        }
        return false;
    }
    
    @Override
    public int hashCode(){
        if (hashCode != Integer.MIN_VALUE)
            return hashCode;
        return (hashCode = name.hashCode());
    }
    
    /**
     * Clears the cache of variables. This method does not have to be called because
     * the caching mechanism uses soft-references therefore the cache should be cleared
     * automatically by the garbage collector if neccessary.
     */
    public static void clearCache(){
        cache.clear();
    }
}
