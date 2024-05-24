/*
 * Constant.java
 *
 * Created on 30. listopad 2006, 16:36
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.ilp.basic;

import java.util.WeakHashMap;

/**
 * Class for representing logical constants. It uses caching so if there are constants with long names
 * which appear in many literals or even in many clauses, the consumed memory will not be too big.
 * 
 * @author Ondra
 */
public class Constant implements Term {
    
    private String name;
    
    private int hashCode = Integer.MIN_VALUE;
    
    private static WeakHashMap<String,Constant> cache = new WeakHashMap<String,Constant>();
    
    /** Creates a new instance of Constant */
    private Constant(String name) {
        this.name = name.trim().intern();
    }
    
    /**
     * Creates an instance of class Constant. before constructing a new instance,
     * it checks if it had not been constructed yet - if it had been constructed then
     * it returns the old instance, otherwise it returns a new instance.
     * @param name
     * @return
     */
    public static Constant construct(String name){
        Constant retVal = cache.get(name);
        if (retVal == null){
            retVal = new Constant(name);
            cache.put(name, retVal);
        }
        return retVal;
    }
    
    /**
     * 
     * @return string representation of the constant
     */
    public String name(){
        return name;
    }
    
    @Override
    public String toString(){
        return name;
    }

    @Override
    public boolean equals(Object o){
        if (o instanceof Constant){
            return o == this || ((Constant)o).name.equals(this.name);
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
     * Clears the cache of Constants - the cache is implemented using soft-references so
     * it is not necessary to call this method manually but it may be useful at some occasions.
     */
    public static void clearCache(){
        cache.clear();
    }
}
