/*
 * Quadruple.java
 *
 * Created on 30. listopad 2006, 20:40
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.utils.tuples;


/**
 * Class representing 4-tuples of objects.
 * @param <R> type of the first object
 * @param <S> type of the second object
 * @param <T> type of the third object
 * @param <U> type of the fourth object
 * 
 * @author Ondra
 */
public class Quadruple<R, S, T, U> {
    
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
    
    /** Creates a new instance of Pair */
    public Quadruple() {
    }
    
    /**
     * Creates a new instance of class Quadruple with the given objects.
     * @param r the first object
     * @param s the second object
     * @param t the third object
     * @param u the fourth object
     */
    public Quadruple(R r, S s, T t, U u){
        this.r = r;
        this.s = s;
        this.t = t;
        this.u = u;
    }

    /**
     * Sets the objects in the Quadruple.
     * @param r the first object
     * @param s the second object
     * @param t the third object
     * @param u the fourth object
     */
    public void set(R r, S s, T t, U u){
        this.r = r;
        this.s = s;
        this.t = t;
        this.u = u;
    }
    
    @Override
    public boolean equals(Object o){
        if (o instanceof Quadruple){
            Quadruple q = (Quadruple)o;
            return q.r.equals(this.r) && q.s.equals(this.s) && q.t.equals(this.t) && q.u.equals(this.u);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return this.r.hashCode()+this.s.hashCode()+this.t.hashCode()+this.u.hashCode();
    }
    
    @Override
    public String toString(){
        return "["+r+", "+s+", "+t+", "+u+"]";
    }
}
