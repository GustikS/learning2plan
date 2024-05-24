/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils.collections;

import java.util.Arrays;
import java.math.BigInteger;
import java.util.Collection;
import java.util.List;
import ida.utils.Sugar;

/**
 * Class for representing fixed-size vectors of instances of class BigInteger.
 * 
 * @author Ondra
 */
public class BigIntegerVector {

    private BigInteger[] values;

    /**
     * Creates a new empty instance of class BigIntegerVector
     */
    public BigIntegerVector(){
        this.values = new BigInteger[0];
    }

    /**
     * Creates a new instance of class BigIntegerVector
     * @param array the array of BigIntegers to be stored in this BigIntegerVector
     */
    public BigIntegerVector(BigInteger[] array){
        this.values = array;
    }

    /**
     * Computes sum of elements in this BigIntegerVector.
     * @return the sum of elements in this BigIntegerVector
     */
    public BigInteger sum(){
        BigInteger retVal = BigInteger.ZERO;
        for (BigInteger bi : values){
            retVal = retVal.add(bi);
        }
        return retVal;
    }

    /**
     * Computes product of elements in this BigIntegerVector.
     * @return the product of elements in this BigIntegerVector
     */
    public BigInteger product(){
        BigInteger retVal = BigInteger.ONE;
        for (BigInteger bi : values){
            retVal = retVal.multiply(bi);
        }
        return retVal;
    }

    /**
     * Computes the sum of two BigIntegerVectors. The result is a BigIntegerVector with
     * the same size as <em>a</em> and <em>b</em>
     * @param a the first BigIntegerVector
     * @param b the second BigIntegerVector
     * @return the sum of two BigIntegerVectors
     */
    public static BigIntegerVector plus(BigIntegerVector a, BigIntegerVector b){
        BigInteger[] retVal = new BigInteger[a.values.length];
        for (int i = 0; i < retVal.length; i++){
            retVal[i] = a.values[i].add(b.values[i]);
        }
        return new BigIntegerVector(retVal);
    }

    /**
     * Creates a new BigIntegerVector by adding the number <em>b</em> to every element
     * of the given BigIntegerVector <em>a</em>.
     * @param a the BigIntegerVector
     * @param b the number
     * @return the sum of the given BigIntegerGVector and the given number
     */
    public static BigIntegerVector add(BigIntegerVector a, BigInteger b){
        BigInteger[] retVal = new BigInteger[a.values.length];
        for (int i = 0; i < retVal.length; i++){
            retVal[i] = a.values[i].add(b);
        }
        return new BigIntegerVector(retVal);
    }

    /**
     * Creates a new BigIntegerVector by subtracting the number <em>b</em> from every element
     * of the given BigIntegerVector <em>a</em>.
     * @param a the BigIntegerVector
     * @param b the number
     * @return the difference of the given BigIntegerGVector and the given number
     */
    public static BigIntegerVector subtract(BigIntegerVector a, BigInteger b){
        BigInteger[] retVal = new BigInteger[a.values.length];
        for (int i = 0; i < retVal.length; i++){
            retVal[i] = a.values[i].subtract(b);
        }
        return new BigIntegerVector(retVal);
    }

    /**
     * Creates a new BigIntegerVector by multiplying the number <em>b</em> with every element
     * of the given BigIntegerVector <em>a</em>.
     * @param a the BigIntegerVector
     * @param b the number
     * @return the product of the given BigIntegerGVector and the given number
     */
    public static BigIntegerVector multiply(BigIntegerVector a, BigInteger b){
        BigInteger[] retVal = new BigInteger[a.values.length];
        for (int i = 0; i < retVal.length; i++){
            retVal[i] = a.values[i].multiply(b);
        }
        return new BigIntegerVector(retVal);
    }

    /**
     * Computes the product of two BigIntegerVectors. The result is a BigIntegerVector with
     * the same size as <em>a</em> and <em>b</em> with elements being products of the respective numbers in
     * the BigIntegerVectors.
     * @param a the first BigIntegerVector
     * @param b the second BigIntegerVector
     * @return the product of two BigIntegerVectors
     */
    public static BigIntegerVector times(BigIntegerVector a, BigIntegerVector b){
        BigInteger[] retVal = new BigInteger[a.values.length];
        for (int i = 0; i < retVal.length; i++){
            retVal[i] = a.values[i].multiply(b.values[i]);
        }
        return new BigIntegerVector(retVal);
    }

    /**
     * Computes sum of the given BigIntegerVectors.
     * @param vectors the BigIntegerVectors (they should all have the same sizes)
     * @return the sum of the BigIntegerVector
     */
    public static BigIntegerVector sum(Collection<BigIntegerVector> vectors){
        BigIntegerVector retVal = null;
        for (BigIntegerVector vector : vectors){
            if (retVal == null){
                retVal = vector;
            } else {
                retVal = plus(retVal, vector);
            }
        }
        return retVal;
    }

    /**
     * Computes product of the given BigIntegerVectors.
     * @param vectors the BigIntegerVectors (they should all have the same sizes)
     * @return the product of the BigIntegerVector
     */
    public static BigIntegerVector product(Collection<BigIntegerVector> vectors){
        BigIntegerVector retVal = null;
        for (BigIntegerVector vector : vectors){
            if (retVal == null){
                retVal = vector;
            } else {
                retVal = times(retVal, vector);
            }
        }
        return retVal;
    }

    /**
     * Creates a new BigIntegerVector in which all occurrences of the given BigInteger are removed.
     * @param vector the original BigIntegerVector
     * @param toBeRemoved the BigInteger to be removed
     * @return the new BigIntegerVector in which all occurrences of the given BigInteger are removed
     */
    public static BigIntegerVector remove(BigIntegerVector vector, BigInteger toBeRemoved){
        return new BigIntegerVector(remove(vector.values, toBeRemoved));
    }

    private static BigInteger[] remove(BigInteger[] vector, BigInteger toBeRemoved){
        BigInteger[] newVector = new BigInteger[vector.length-Sugar.countOccurences(toBeRemoved, vector)];
        int index = 0;
        for (BigInteger bi : vector){
            if (!bi.equals(toBeRemoved)){
                newVector[index] = bi;
                index++;
            }
        }
        return newVector;
    }

    /**
     * 
     * @return the BigIntegers contained in this object
     */
    public BigInteger[] values(){
        return values;
    }
    
    @Override
    public String toString(){
        return Sugar.objectArrayToString(values);
    }

    @Override
    public int hashCode(){
        return Arrays.hashCode(values);
    }

    @Override
    public boolean equals(Object o){
        if (o instanceof BigIntegerVector){
            BigIntegerVector ia = (BigIntegerVector)o;
            if (Arrays.equals(ia.values, this.values)){
                return true;
            }
        }
        return false;
    }
}
