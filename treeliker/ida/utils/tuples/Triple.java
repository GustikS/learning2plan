/*
 * Triple.java
 *
 * Created on 6. prosinec 2006, 19:19
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.utils.tuples;

/**
 * Class representing 3-tuples of objects.
 * @param <R> type of the first object
 * @param <S> type of the second object
 * @param <T> type of the third object
 * 
 * @author Ondra
 */
public class Triple<R, S, T> {
    
    /**
     * The first object
     */
    public R r;
    
    /**
     * The second object
     */
    public S s;
    
    /**
     * The third object
     */
    public T t;
    
    /** Creates a new instance of Pair */
    public Triple() {
    }
    
    /**
     * Creates a new instance of class Triple with the given objects.
     * @param r the first object
     * @param s the second object
     * @param t the third object
     */
    public Triple(R r, S s, T t){
        this.r = r;
        this.s = s;
        this.t = t;
    }

    /**
     * Sets the objects in the Triple.
     * @param r the first object
     * @param s the second object
     * @param t the third object
     */
    public void set(R r, S s, T t){
        this.r = r;
        this.s = s;
        this.t = t;
    }

    @Override
    public boolean equals(Object o){
        if (o instanceof Triple){
            Triple t = (Triple)o;
            return ((t.r == null && this.r == null) || (t.r != null && this.r != null && t.r.equals(this.r))) &&
                    ((t.s == null && this.s == null) || (t.s != null && this.s != null && t.s.equals(this.s))) &&
                    ((t.t == null && this.t == null) || (t.t != null && this.t != null && t.t.equals(this.t)));
        }
        return false;
    }

    @Override
    public int hashCode() {
        int hashCode = 0;
        if (this.r != null){
            hashCode += this.r.hashCode();
        }
        if (this.s != null){
            hashCode += this.s.hashCode();
        }
        if (this.t != null){
            hashCode += this.t.hashCode();
        }
        return hashCode;
    }
    
    @Override
    public String toString(){
        return "["+r+", "+s+", "+t+"]";
    }
}

