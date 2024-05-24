/*
 * Combinatorics.java
 *
 * Created on 11. duben 2007, 13:01
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.utils;

import java.util.*;
import ida.utils.tuples.Tuple;

/**
 * Class providing several useful combinatorics operations.
 * 
 * @author Ondra
 */
public class Combinatorics {
    
    private static Random random = new Random();
    
    private Combinatorics() {
    }
    
    /**
     * Creates a random combination of <em>k</em> elements from the given <em>list</em>
     * @param <T> type of the elenments in the list
     * @param list the list from which the elements should be selected
     * @param k number of elements in the combination
     * @return random combination of <em>k</em> elements from <em>list</em>
     */
    public static <T> Tuple<T> randomCombination(List<T> list, int k){
        return randomCombination(list, k, random);
    }
    
    /**
     * 
     * Creates a random combination of <em>k</em> elements from the given <em>list</em>
     * @param <T> type of the elenments in the list
     * @param list the list from which the elements should be selected
     * @param k number of elements in the combination
     * @param rand random number generator which should be used to select the random combination
     * @return random combination of <em>k</em> elements from <em>list</em>
     */
    public static <T> Tuple<T> randomCombination(List<T> list, int k, Random rand){
        if (list.size() < k){
            throw new IllegalArgumentException("Illegal arguments: list.size() < k.");
        }
        if (k == 0){
            return new Tuple<T>(0);
        }
        Set<Integer> s = new HashSet<Integer>();
        int n = list.size();
        for (int j = n-k+1; j <= n; j++){
            int t = rand.nextInt(j)+1;
            if (!s.contains(t)){
                s.add(t);
            } else {
                s.add(j);
            }
        }
        Tuple<T> t = new Tuple<T>(k);
        int index = 0;
        for (Integer i : s){
            t.set(list.get(i-1), index);
            index++;
        }
        return t;
    }
    
    /**
     * Creates a list which contains all sub-sequences of the given <em>list</em>.
     * The sub-sequences preserve the order of the elements in the original list.
     * @param <T> the type of the elements in the list (and in the resulting sub-sequences)
     * @param list the list from which the sub-sequences should be generated
     * @return list of all sub-sequences (sub-sequences are represented as instances
     * of class Tuple<T>)
     */
    public static <T> List<Tuple<T>> allSubsequences(List<T> list){
        ArrayList<Tuple<T>> ret = new ArrayList<Tuple<T>>(1 << list.size());
        int listsize = list.size();
        ArrayList temp = new ArrayList();
        for (int i = 0; i < 1 << listsize; i++){
            for (int j = 0; j < listsize; j++){
                if ((i / (1 << j)) % 2 == 0)
                    temp.add(list.get(j));
            }
            ret.add(new Tuple(temp));
            temp.clear();
        }
        return ret;
    }
    
    /**
     * Creates a list which contains all sub-sequences of length <em>k</em> from the given <em>list</em>.
     * The sub-sequences preserve the order of the elements in the original list.
     * @param <T> the type of the elements in the list (and in the resulting sub-sequences)
     * @param list the list from which the sub-sequences should be generated
     * @param k 
     * @return list of all sub-sequences (sub-sequences are represented as instances
     * of class Tuple<T>)
     */
    public static <T> List<Tuple<T>> allSubsequences(List<T> list, int k){
        List<Tuple<T>> retVal = new ArrayList<Tuple<T>>();
        List<int[]> ints = new ArrayList<int[]>();
        int n = list.size();
        for (int i = 0; i < k; i++){
            ints = allNextSubsequences(ints, n);
        }
        for (int[] i : ints){
            Tuple t = new Tuple(i.length);
            for (int j = 0; j < i.length; j++){
                t.set(list.get(i[j]), j);
            }
            retVal.add(t);
        }
        return retVal;
    }
    
    private static List<int[]> allNextSubsequences(List<int[]> list, int n){
        List<int[]> tuples = new ArrayList<int[]>();
        if (list.isEmpty()){
            for (int i = 0; i < n; i++){
                int[] tuple = new int[1];
                tuple[0] = i;
                tuples.add(tuple);
            }
        } else {
            for (int i = 0; i < list.size(); i++){
                int[] oldCombination = list.get(i);
                for (int j = oldCombination[oldCombination.length-1]+1; j < n; j++){
                    int[] newCombination = new int[oldCombination.length+1];
                    System.arraycopy(oldCombination, 0, newCombination, 0, oldCombination.length);
                    newCombination[oldCombination.length] = j;
                    tuples.add(newCombination);
                }
            }
        }
        return tuples;
    }
    
    /**
     * Computes factorial of the given number
     * @param n the number
     * @return <em>n!</em>
     */
    public static double factorial(int n){
        if (n < 0)
            return -1;
        double fact = 1;
        for (int i = 1; i <= n; i++){
            fact *= i;
        }
        return fact;
    }
    
    /**
     * Computes logarithm of factorial of the given number
     * @param n the number
     * @return <em>ln(n!)</em>
     */
    public static double logFactorial(int n){
        if (n < 0)
            return Double.NaN;
        double fact = 0;
        for (int i = 1; i <= n; i++){
            fact += Math.log(i);
        }
        return fact;
    }
    
    /**
     * Computes the value of the binomial number "<em>n</em> over <em>k</em>".
     * @param n the number <em>n</em>
     * @param k the number <em>k</em>
     * @return the value of the binomial number "<em>n</em> over <em>k</em>"
     */
    public static double binomial(int n, int k){
        return factorial(n)/(factorial(k)*factorial(n-k));
    }
    
    /**
     * Computes value of the logarithm of the binomial number "<em>n</em> over <em>k</em>".
     * @param n the number <em>n</em>
     * @param k the number <em>k</em>
     * @return the value of the binomial number ln("<em>n</em> over <em>k</em>")
     */
    public static double logBinomial(int n, int k){
        return logFactorial(n) - logFactorial(k) - logFactorial(n-k);
    }
}
