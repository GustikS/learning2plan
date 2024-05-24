/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils.collections;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import ida.utils.Sugar;
/**
 *
 * Class for datastructure which roughly coprresponds to java.util.Map<R,java.util.List<S>>.
 * 
 * @param <R> type of key-elements
 * @param <S> type of value-elements
 * @author Ondra
 */
public class MultiList<R,S> {

    public final static int ARRAY_LIST = 1, LINKED_LIST = 2;
    
    private int listType = ARRAY_LIST;
    
    private List<S> emptyList = new ArrayList<S>();
    
    private ConcurrentHashMap<R,List<S>> map = new ConcurrentHashMap<R,List<S>>();

    /**
     * Creates a new instance of class MultiList
     */
    public MultiList(){}
    
    /**
     * Creates a new instance of class MultiList
     * @param listType type of list used in the datastructure - can be MultiList.ARRAY_LIST or MultiList.LINKED_LIST
     */
    public MultiList(int listType){
        this.listType = listType;
    }
    
    /**
     * 
     * @return number of different key-elements
     */
    public int size() {
        return map.size();
    }

    /**
     * 
     * @return true if the MultiList is empty
     */
    public boolean isEmpty() {
        return map.isEmpty();
    }

    /**
     * Checks if the MultiList contains the given key.
     * @param key the key
     * @return true if the MultiList contains the given key
     */
    public boolean containsKey(Object key) {
        return map.containsKey(key);
    }

    /**
     * Returns a list of elements associated to <em>key</em>. If there is no list associated to <em>key</em> then
     * an empty list is returned.
     * @param key the key
     * @return list of elements associated to <em>key</em>. If there is no list associated to <em>key</em> then
     * an empty list is returned.
     */
    public List<S> get(Object key) {
        if (map.containsKey(key))
            return map.get(key);
        else
            return emptyList;
    }

    /**
     * Adds the key-value pair to the MultiList. It does not matter whether the MultiList already contains
     * this key-value pair, the value will be simply added to the list associated with the given key.
     * @param key the key
     * @param value the value
     */
    public void put(R key, S value) {
        if (!map.containsKey(key)){
            if (listType == ARRAY_LIST){
                map.put(key, Collections.synchronizedList(new ArrayList<S>()));
            } else if (listType == LINKED_LIST){
                map.put(key, Collections.synchronizedList(new LinkedList<S>()));
            }
        }
        map.get(key).add(value);
    }

    /**
     * Adds all the key-value pair to the MultiList. It does not matter whether the MultiList already contains
     * any of the key-value pairs, the values will be simply added to the list associated with the given key.
     * @param key the key
     * @param value the collection of values
     */
    public void putAll(R key, Collection<S> values){
        if (!map.containsKey(key)){
            if (listType == ARRAY_LIST){
                map.put(key, new ArrayList<S>());
            } else if (listType == LINKED_LIST){
                map.put(key, new LinkedList<S>());
            }
        }
        map.get(key).addAll(values);
    }

    /**
     * Sets the values associated with the given key.
     * @param key the key
     * @param value the new values
     */
    public void set(R key, List<S> value){
        map.remove(key);
        map.put(key, value);
    }

    /**
     * Sets the values associated with the given key.
     * @param key the key
     * @param value the new values
     */
    public void set(R key, Collection<S> value){
        map.remove(key);
        map.put(key, Collections.synchronizedList(Sugar.listFromCollections(value)));
    }

    /**
     * Removes the first occurrence of the given value from the list associated with the given key.
     * @param key the key
     * @param value the value to be removed
     */
    public void remove(Object key, Object value) {
        map.get(key).remove(value);
    }
    
    /**
     * Removes all values associated with the given key.
     * @param key the key
     */
    public void remove(Object key){
        map.remove(key);
    }
    
    /**
     * Removes everything from the MultiList.
     */
    public void clear() {
        map.clear();
    }

    /**
     * 
     * @return the set of all key-elements
     */
    public Set<R> keySet() {
        return map.keySet();
    }

    /**
     * 
     * @return a collection containing all the values
     */
    public Collection<List<S>> values(){
        return map.values();
    }

    /**
     * 
     * @return the backing entry-set 
     */
    public Set<Entry<R, List<S>>> entrySet(){
        return map.entrySet();
    }
    
    @Override
    public String toString(){
        return this.map.toString();
    }

    /**
     * 
     * @return string with the numbers of elements associated to particular keys
     */
    public String sizesToString(){
        StringBuilder sb = new StringBuilder();
        sb.append("HashListBag[");
        int index = 0;
        for (Map.Entry<R,List<S>> entry : this.map.entrySet()){
            sb.append(entry.getKey()).append(" ~ ").append(entry.getValue().size());
            if (index++ < this.map.size()-1){
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * 
     * @return int[] array with numbers of elements associated to particular keys
     */
    public int[] sizes(){
        int[] sizes = new int[this.map.size()];
        int i = 0;
        for (Map.Entry<R,List<S>> entry : map.entrySet()){
            sizes[i] = entry.getValue().size();
            i++;
        }
        return sizes;
    }
    
    /**
     * Creates a copy of the MultiList
     * @return a copy of the MultiList
     */
    public MultiList<R,S> copy(){
        MultiList<R,S> retVal = new MultiList<R,S>();
        retVal.putAll_forCopy(this);
        return retVal;
    }

    private void putAll_forCopy(MultiList<R,S> bag){
        for (Map.Entry<R,List<S>> entry : bag.entrySet()){
            this.putAll(entry.getKey(), ida.utils.Sugar.listFromCollections(entry.getValue()));
        }
    }
}
