/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils.collections;

import java.util.*;

/**
 * Class for converting objects to unique identifiers (integers) and back.
 * 
 * @param <T> type of the objects
 * @author Ondra
 */
public class ValueToIndex<T> {
    
    private int lastIndex = 0;
    
    private LinkedHashMap<T,Integer> valueToIndex = new LinkedHashMap<T,Integer>();
    
    private LinkedHashMap<Integer,T> indexToValue = new LinkedHashMap<Integer,T>();

    /**
     * Converts the given object <em>t</em> to a unique integer.
     * @param t the object
     * @return the unique integer representing the object
     */
    public int valueToIndex(T t){
        if (!valueToIndex.containsKey(t)){
            valueToIndex.put(t, lastIndex);
            indexToValue.put(lastIndex, t);
            lastIndex++;
        }
        return valueToIndex.get(t);
    }
    
    /**
     * Converts the given unique identifier back to the original object (the method
     * valueToIndex with this object must have had been called prior to calling this method,
     * otherwise the method would return null).
     * @param index the unique identifier of the object
     * @return the object corresponding to the given unique identifier or null if there is no such object
     */
    public T indexToValue(int index){
        if (!indexToValue.containsKey(index))
            return null;
        else
            return indexToValue.get(index);
    }
   
    
    /**
     * Adds the given pair unique identifier - object
     * @param key the unique identifier
     * @param value the object
     */
    public void put(int key, T value){
        this.valueToIndex.put(value, key);
        this.indexToValue.put(key, value);
    }
    
    /**
     * Creates a set of unique identifiers for the objects in the given collection.
     * @param coll the collection of objects
     * @return the set of unique identifiers for the objects in the collection <em>coll</em>
     */
    public Set<Integer> valuesToIndices(Collection<T> coll){
        Set<Integer> retVal = new HashSet<Integer>();
        for (T t : coll){
            retVal.add(valueToIndex(t));
        }
        return retVal;
    }
    
    /**
     * Creates a set of objects for the unique identifiers in the given collection.
     * @param coll the collection of unique identifiers
     * @return the set of objects for the unique identifiers in the collection <em>coll</em>
     */
    public Set<T> indicesToValues(Collection<Integer> coll){
        Set<T> retVal = new HashSet<T>();
        for (Integer i : coll){
            retVal.add(indexToValue(i));
        }
        return retVal;
    }
    
    /**
     * 
     * @return number of elements for which there are the unique IDs
     */
    public int size(){
        return this.valueToIndex.size();
    }
    
    /**
     * 
     * @return the objects for which there are the unique IDs
     */
    public Set<T> values(){
        return valueToIndex.keySet();
    }
    
    /**
     * 
     * @return the unique IDs
     */
    public Set<Integer> indices(){
        return indexToValue.keySet();
    }
    
    @Override
    public String toString(){
        return this.valueToIndex.toString();
    }
}
