/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ida.utils;

import java.lang.ref.SoftReference;
import java.util.HashMap;

/**
 * Class implementing a simple cache.
 * 
 * @param <R> type of the keys of the cached values
 * @param <S> type of the cached values
 * @author Ondra
 */
public class Cache<R,S> {
    
    private SoftReference<HashMap<R,S>> softRef;
    
    /**
     * Creates a new empty instance of class Cache
     */
    public Cache(){}
    
    /**
     * Stores a new element to the cache.
     * @param key key of the element
     * @param value the lement to be stored
     */
    public void put(R key, S value){
        HashMap<R,S> map = null;
        if (softRef == null || (map = softRef.get()) == null){
            map = new HashMap<R,S>();
            softRef = new SoftReference<HashMap<R,S>>(map);
        }
        map.put(key, value);
    }
    
    /**
     * Retrieves an element from this cache.
     * @param key key of the element
     * @return the element associated to the key or null
     */
    public S get(R key){
        HashMap<R,S> map = null;
        if (softRef == null || (map = softRef.get()) == null){
            return null;
        }
        return map.get(key);
    }
    
    /**
     * Clears this cache
     */
    public void clear(){
        if (this.softRef != null){
            this.softRef.clear();
        }
    }
}
