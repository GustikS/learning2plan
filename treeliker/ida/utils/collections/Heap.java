/*
 * Heap.java
 *
 * Created on 6. květen 2007, 11:14
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.utils.collections;

import java.util.*;
import ida.utils.Sugar;
import ida.utils.VectorUtils;

import ida.utils.tuples.*;
/**
 * Priority front implemnted as Heap.
 * 
 * @param <T> type of elements in the Heap
 * @author Ondra
 */
public class Heap<T> {
    
    private double keys[];
    
    private T values[];
    
    private int last = 0;
    
    private double growFactor = 2;
    
    //statistics
    private int maxSize = 0;
    
    /** Creates a new instance of Heap */
    public Heap() {
        this(32);
    }
    
    /**
     * Creates a new instance of class Heap with the given initial capacity
     * @param initialCapacity the initial capacity
     */
    public Heap(int initialCapacity){
        this.keys = new double[initialCapacity];
        this.keys[0] = Integer.MIN_VALUE;
        this.values = (T[])new Object[initialCapacity];
    }

    /**
     * Removes the minimal element and returns it (if there is more than one such elements, it
     * selects one of them).
     * @return the minimal element
     */
    public T removeMin(){
        if (last > 1){
            T ret = values[1];
            swap(1, last);
            last--;
            repairTopDown(1);
            return ret;
        }
        else if (last == 1){
            last = 0;
            return values[1];
        }
        else
            throw new java.lang.IllegalStateException();
    }
    
    /**
     * Returns the minimal element (if there is more than one such elements, it
     * selects one of them). It does not remove the returned element from the Heap.
     * @return the minimal element
     */
    public T lookAtMinValue(){
        return values[1];
    }
    
    /**
     * 
     * @return the "value" of the minimal element
     */
    public double lookAtMinKey(){
        return keys[1];
    }
    
    /**
     * 
     * @return true if the heap is nonEmpty
     */
    public boolean hasNext(){
        return last != 0;
    }
    
    /**
     * 
     * @return number of elements in the Heap
     */
    public int size(){
        return last;
    }
    
    /**
     * Adds new element to the heap. If there is already an element with the same key,
     * then this new element is still added - there may be more than one element with the same key 
     * in a Heap.
     * @param key the key (=priority)
     * @param value the value to be stored
     */
    public void add(double key, T value){
        last++;
        if (last == keys.length)
            realloc();
        values[last] = value;
        keys[last] = key;
        repairBottomUp(last);
        //
        this.maxSize = Math.max(maxSize, last-1);
    }
    
    private void repairBottomUp(int leaf){
        while (leaf > 1 && keys[leaf] < keys[leaf/2]){
            swap(leaf, leaf/2);
            leaf /= 2;
        }
    }
    
    private void repairTopDown(int root){
        while (true){
            if (2*root+1 <= last && keys[2*root+1] < keys[2*root] && keys[root] > keys[2*root+1]){
                swap(root, 2*root+1);
                root = 2*root+1;
            }
            else if (2*root <= last && keys[root] > keys[2*root]){
                swap(root, 2*root);
                root *= 2;
            }
            else
                break;
        }
    }
    
    private void swap(int a, int b){
        double temp = keys[a];
        keys[a] = keys[b];
        keys[b] = temp;
        T ttemp = values[a];
        values[a] = values[b];
        values[b] = ttemp;
    }
    
    private void realloc(){
        double[] newKeys = new double[(int)(keys.length*growFactor)];
        System.arraycopy(keys, 0, newKeys, 0, keys.length);
        this.keys = newKeys;
        T[] newValues = (T[])new Object[(int)(values.length*growFactor)];
        System.arraycopy(values, 0, newValues, 0, values.length);
        this.values = newValues;
    }
    
    /**
     * 
     * @return the list of pairs: [key, value] which are contained in the Heap
     */
    public List<Pair<T,Double>> values(){
        List<Pair<T,Double>> list = new ArrayList<Pair<T,Double>>();
        for (int i = 1; i <= this.last; i++){
            list.add(new Pair<T,Double>(this.values[i], this.keys[i]));
        }
        return list;
    }

    @Override
    public String toString(){
        return "Heap.values: "+Sugar.objectArrayToString(this.values)+", Heap.keys: "+VectorUtils.doubleArrayToString(keys);
    }

    @Override
    public int hashCode(){
        return Arrays.hashCode(this.values)+Arrays.hashCode(this.keys);
    }

    @Override
    public boolean equals(Object o){
        if (o instanceof Heap){
            Heap h = (Heap)o;
            return Arrays.equals(this.values, h.values) && Arrays.equals(this.keys, h.keys);
        }
        return false;
    }

//    public static void main(String args[]){
//        Heap<Integer> h = new Heap<Integer>();
//        for (int i = 0; i < 20; i++){
//            h.add((int)(2*Math.random()), i);
//        }
//        while (h.hasNext()){
//            System.out.println(h.lookAtMinKey()+" "+h.removeMin());
//        }
//    }
}
