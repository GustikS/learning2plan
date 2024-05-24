/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils.collections;

import java.util.*;
import ida.utils.tuples.Pair;
/**
 * Class for tracking counts of objects. 
 * 
 * @param <T> type of the objects for which counts should be tracked
 * @author Ondra
 */
public class Counters<T> {
    
    private HashMap<T,Integer> map = new HashMap<T,Integer>();

    /**
     * Creates a new empty instance of class BigIntegerCounts
     */
    public Counters(){}

    /**
     * Creates new instance of class Counters<T> initialized with the counts given
     * as a set of pairs: [key, count]
     * @param <T> the type of the counted objects
     * @param pairs the initial counts specified as pairs: [key, count]
     * @return new instance of class Counters<T> initialized with the counts given
     * as a set of pairs: [key, count]
     */
    public static <T> Counters<T> createCounters(Set<Pair<T,Integer>> pairs){
        Counters<T> counters = new Counters<T>();
        for (Pair<T,Integer> pair : pairs){
            counters.add(pair.r, pair.s);
        }
        return counters;
    }

    /**
     * Adds <em>value</em> to the count associated to the element <em>key</em>
     * @param key the element for which count should be modified
     * @param value the increment by which the value should be increased
     */
    public void add(T key, int value){
        addPre(key, value);
    }

    /**
     * Adds <em>value</em> to the count associated to the element <em>key</em>
     * @param key the element for which count should be modified
     * @param value the increment by which the value should be increased
     * @return the count associated to <em>key</em> after the modification
     */
    public int addPre(T key, int value){
        if (!map.containsKey(key)){
            map.put(key, value);
        } else {
            map.put(key, map.get(key)+value);
        }
        return map.get(key);
    }

    /**
     * Adds <em>value</em> to the count associated to the element <em>key</em>
     * @param key the element for which count should be modified
     * @param value the increment by which the value should be increased
     * @return the count associated to <em>key</em> before the modification
     */
    public int addPost(T key, int value){
        int oldValue = 0;
        if (!map.containsKey(key)){
            map.put(key, value);
        } else {
            oldValue = map.get(key);
            map.put(key, map.get(key)+value);
        }
        return oldValue;
    }

    /**
     * Increments by 1 the count associated to the element <em>key</em>
     * @param key the element for which count should be incremented
     */
    public void increment(T key){
        this.incrementPre(key);
    }
    
    /**
     * Decrements by 1 the count associated to the element <em>key</em>
     * @param key the element for which count should be decremented
     */
    public void decrement(T key){
        this.decrementPre(key);
    }
    
    /**
     * 
     * @param key the element for which we want to get the value
     * @return the value associated to element <em>key</em>
     */
    public int get(T key){
        if (!map.containsKey(key)){
            return 0;
        } else {
            return map.get(key);
        }
    }
    
    /**
     * Increments the count associated to the element <em>key</em> by 1.
     * @param key the element for which count should be modified
     * @param value the increment by which the value should be increased
     * @return the count associated to <em>key</em> after the modification
     */
    public int incrementPre(T key){
        if (!map.containsKey(key)){
            map.put(key, 1);
        } else {
            map.put(key, map.get(key)+1);
        }
        return map.get(key);
    }
    
    /**
     * Increments the count associated to the element <em>key</em> by 1.
     * @param key the element for which count should be modified
     * @return the count associated to <em>key</em> before the modification
     */
    public int incrementPost(T key){
        if (!map.containsKey(key)){
            map.put(key, 1);
        } else {
            map.put(key, map.get(key)+1);
        }
        return map.get(key)-1;
    }
    
    /**
     * Decrements the count associated to the element <em>key</em> by 1.
     * @param key the element for which count should be modified
     * @return the count associated to <em>key</em> after the modification
     */
    public int decrementPre(T key){
        if (!map.containsKey(key)){
            map.put(key, -1);
        } else {
            map.put(key, map.get(key)-1);
        }
        return map.get(key);
    }
    
    /**
     * Decrements the count associated to the element <em>key</em> by 1.
     * @param key the element for which count should be modified
     * @return the count associated to <em>key</em> before the modification
     */
    public int decrementPost(T key){
        if (!map.containsKey(key)){
            map.put(key, -1);
        } else {
            map.put(key, map.get(key)-1);
        }
        return map.get(key)+1;
    }
    
    /**
     * 
     * @return all elements for which counts are tracked
     */
    public Set<T> keySet(){
        return this.map.keySet();
    }

    /**
     * 
     * @return the counts of all keys in one collection
     */
    public Collection<Integer> counts(){
        return this.map.values();
    }

    /**
     * Adds all keys and values of another instance of class BigIntegerCounters<T>.
     * @param counters the instance of class BigIntegerCounters<T> whose content should be added
     */
    public void addAll(Counters<T> counters){
        for (Map.Entry<T,Integer> entry : counters.map.entrySet()){
            if (this.map.containsKey(entry.getKey())){
                this.map.put(entry.getKey(), this.map.get(entry.getKey())+entry.getValue());
            } else {
                this.map.put(entry.getKey(), entry.getValue());
            }
        }
    }

    @Override
    public String toString(){
        return this.map.toString();
    }

    @Override
    public int hashCode(){
        return this.map.hashCode();
    }

    @Override
    public boolean equals(Object o){
        if (o instanceof Counters){
            return ((Counters)o).map.equals(this.map);
        }
        return false;
    }

    /**
     * 
     * @return map in the following form: element -> count
     */
    public Map<T,Integer> toMap(){
        HashMap<T,Integer> copy = new HashMap<T,Integer>();
        copy.putAll(this.map);
        return copy;
    }

    /**
     * 
     * @return number of alements for which counts are tracked
     */
    public int size(){
        return this.map.size();
    }
}
