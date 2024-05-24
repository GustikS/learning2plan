/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.ilp.basic;

import ida.utils.tuples.*;

import java.util.*;
/**
 * Class representing Prolog lists.
 * 
 * @author Ondra
 */
public class PrologList implements Term {

    private int hashCode = -1;
    
    private Term items[];
    
    private String toString;
    
    /**
     * Creates a new instance of class PrologList
     * @param itemsCount number of elements in the list
     */
    public PrologList(int itemsCount){
        this.items = new Term[itemsCount];
    }
    
    /**
     * Creates a new instance of class PrologList
     * @param termList list of terms in the list
     */
    public PrologList(List<Term> termList){
        this(termList.size());
        int index = 0;
        for (Term t : termList){
            this.items[index] = t;
            index++;
        }
    }
    
    /**
     * 
     * @param index
     * @return element of the list at position index
     */
    public Term get(int index){
        return this.items[index];
    }
    
    /**
     * Sets the element at position index in the list
     * @param term the term to be placed at positin index
     * @param index the position where the term should be placed
     */
    public void set(Term term, int index){
        this.items[index] = term;
    }
    
    /**
     * 
     * @return number of items in this list
     */
    public int countItems(){
        return this.items.length;
    }
    
    @Override
    public boolean equals(Object o){
        if (o instanceof PrologList){
            PrologList pl = (PrologList)o;
            return pl.toString().equals(this.toString());
        }
        return false;
    }
    
    @Override
    public int hashCode(){
        if (this.hashCode == -1){
            this.hashCode = this.toString().hashCode();
        }
        return this.hashCode;
    }
    
    @Override
    public String toString(){
        if (this.toString == null){
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int i = 0; i < items.length; i++){
                sb.append(items[i]);
                if (i < items.length-1){
                    sb.append(", ");
                }
            }
            sb.append("]");
            this.toString = sb.toString();
        }
        return this.toString;
    }
    
    /**
     * 
     * @return string representation of this list
     */
    public String name() {
        return this.toString();
    }

    /**
     * Creates a flattened version of the list.
     * @return flattened version of the list.
     */
    public PrologList flatten(){
        List<Term> termList = new ArrayList<Term>();
        flatten(this, termList);
        return new PrologList(termList);
    }
    
    private void flatten(PrologList prologList, List<Term> termList){
        for (Term t : prologList.items){
            if (t instanceof PrologList){
                flatten((PrologList)t, termList);
            } else {
                termList.add(t);
            }
        }
    }
    
    /**
     * Parses the list from its string representation.
     * @param str the string representation of the list
     * @param variables Map of used variables - it does not have to contain the variables contained in the arguments of the list,
     * if they are not there, they will be added automatically
     * @param constants Map of used constants - it does not have to contain the constants contained in the arguments of the list,
     * if they are not there, they will be added automatically
     * @return
     */
    public static PrologList parseList(String str, Map<String,Variable> variables, Map<String,Constant> constants){
        char[] c = str.toCharArray();
        int index = 1;
        ArrayList<Term> items = new ArrayList<Term>();
        while (true){
            if (index >= c.length || c[index] == ']'){
                break;
            } else if (c[index] == ' '){
                //do nothing
            } else {
                Pair<Term,Integer> pair = ParserUtils.parseTerm(c, index, ']', variables, constants);
                items.add(pair.r);
                index = pair.s;
            }
            index++;
        }
        PrologList retVal = new PrologList(items.size());
        for (int i = 0; i < items.size(); i++){
            retVal.set(items.get(i), i);
        }
        return retVal;
    }
}
