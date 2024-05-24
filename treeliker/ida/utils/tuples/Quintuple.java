/*
 * Quintuple.java
 *
 * Created on 13. leden 2008, 11:30
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */


package ida.utils.tuples;

/**
 * Class representing 5-tuples of objects.
 * @param <R> type of the first object
 * @param <S> type of the second object
 * @param <T> type of the third object
 * @param <U> type of the fourth object
 * @param <V> type of the fifth object
 * 
 * @author Ondra
 */
public class Quintuple<R,S,T,U,V> {
    
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
    
    /**
     * The fourth object
     */
    public U u;
    
    /**
     * The fifth object
     */
    public V v;
    
    /** Creates a new instance of Quintuple */
    public Quintuple() {
    }
    
    /**
     * Creates a new instance of class Quintuple with the given objects.
     * @param r the first object
     * @param s the second object
     * @param t the third object
     * @param u the fourth object
     * @param v the fifth object
     */
    public Quintuple(R r, S s, T t, U u, V v){
        this.r = r;
        this.s = s;
        this.t = t;
        this.u = u;
        this.v = v;
    }

    /**
     * Sets the objects in the Quintuple.
     * @param r the first object
     * @param s the second object
     * @param t the third object
     * @param u the fourth object
     * @param v the fifth object
     */
    public void set(R r, S s, T t, U u, V v){
        this.r = r;
        this.s = s;
        this.t = t;
        this.u = u;
        this.v = v;
    }

    @Override
    public boolean equals(Object o){
        if (o instanceof Quintuple){
            Quintuple q = (Quintuple)o;
            return q.r.equals(this.r) && q.s.equals(this.s) && q.t.equals(this.t) && q.u.equals(this.u) && q.v.equals(this.v);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return this.r.hashCode()+this.s.hashCode()+this.t.hashCode()+this.u.hashCode()+this.v.hashCode();
    }
    
    @Override
    public String toString(){
        return "["+r+", "+s+", "+t+", "+u+", "+v+"]";
    }
}
