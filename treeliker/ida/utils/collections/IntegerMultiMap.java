/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils.collections;

import java.util.*;
import java.util.Map.Entry;

/**
 * MultiMap of Integers - it stores sets of integers as instances of class IntegerSet.
 * This means that building an IntegerMultiMap by adding values one by one is not
 * a good idea as it could be very slow (IntegerSets are immutable so every addition
 * of an element to a MultiMap means creating a new IntegerSet which is slow).
 * @param <R> type of key-elements
 * @author Ondra
 */
public class IntegerMultiMap<R> {
    
    private Map<R,IntegerSet> map = new HashMap<R,IntegerSet>();

    /**
     * Creates a new instance of IntegerMulttiMap from an instance of class MultiMap<R,Integer>
     * @param <R> type of key-elements in the new constructed IntegerMultiMap
     * @param multiMap MultiMap from which the IntegerMultiMap should be constructed
     * @return a new instance of IntegerMulttiMap from the given instance of class MultiMap<R,Integer>
     */
    public static <R> IntegerMultiMap createIntegerMultiMap(MultiMap<R,Integer> multiMap){
        IntegerMultiMap<R> ib = new IntegerMultiMap<R>();
        for (Map.Entry<R,Set<Integer>> entry : multiMap.entrySet()){
            ib.add(entry.getKey(), IntegerSet.createIntegerSet(entry.getValue()));
        }
        return ib;
    }
    
    /**
     * 
     * @return number of different key-values in the IntegerMultiMap
     */
    public int size() {
        return map.size();
    }

    /**
     * 
     * @return true if the IntegerMultiMap is empty, false otherwise
     */
    public boolean isEmpty() {
        return map.isEmpty();
    }

    /**
     * Checks if the IntegerMultiMap contains the given key-value (and an associated IntegerSet).
     * @param key the key
     * @return true if the IntegerMultiMap contains the given key-value
     */
    public boolean containsKey(R key) {
        return map.containsKey(key);
    }

    /**
     * 
     * @param key the key
     * @return IntegerKey associated to <em>key</em>. If there is no IntegerSet associated to <em>key</em> then
     * IntegerSet.emptySet is returned.
     */
    public IntegerSet get(R key) {
        if (map.containsKey(key))
            return map.get(key);
        else
            return IntegerSet.emptySet;
    }

    /**
     * Adds a new set of integers in the form of IntegerSet object.
     * @param key the key
     * @param value the value
     */
    public void add(R key, IntegerSet value) {
        if (!map.containsKey(key)){
            map.put(key, value);
        } else {
            map.put(key, IntegerSet.union(value, map.get(key)));
        }
    }
    
    /**
     * Removes the IntegerSet associated to the given key.
     * @param key the key
     */
    public void remove(R key){
        map.remove(key);
    }
    
    /**
     * Clears the IntegerMultiMap.
     */
    public void clear() {
        map.clear();
    }

    /**
     * 
     * @return all key-values contained in the IntegerMultiMap
     */
    public Set<R> keySet() {
        return map.keySet();
    }

    /**
     * 
     * @return all IntegerSets contained in the IntegerMultiMap
     */
    public Collection<IntegerSet> values() {
        return map.values();
    }

    /**
     * 
     * @return the entry-set of the IntegerMultiMap
     */
    public Set<Entry<R, IntegerSet>> entrySet() {
        return map.entrySet();
    }
    
    @Override
    public String toString(){
        return this.map.toString();
    }
}