/*
 * Predicate.java
 *
 * Created on 30. listopad 2006, 16:36
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.ilp.basic;

import ida.utils.UniqueIDs;
import ida.utils.tuples.*;

import java.util.*;
import ida.utils.collections.FakeMap;
/**
 * Class for representing positive (i.e. nonnegated) first-order-logic literals.
 * 
 * @author Ondra
 */
public class Literal {
    
    private Term[] terms;
    
    private String predicate;
    
    private int id;
    
    private int hashCode = -1;
    
    private int nameHashCode;
    
    private boolean changePermitted = true;

    private static Map<String,Variable> fakeMapVar = new FakeMap<String,Variable>();

    private static Map<String,Constant> fakeMapConst = new FakeMap<String,Constant>();
    
    private Literal(){}
    
    /**
     * Creates a new instance of class Literal. When name = "somePredicate" and arity = 3,
     * then the result is Literal somePredicate(...,...,...)
     * 
     * @param name predicate-name of the literal
     * @param arity arity of the literal
     */
    public Literal(String name, int arity) {
        this.id = (int)UniqueIDs.getUniqueName();
        this.terms = new Term[arity];
        this.predicate = name.intern();
        this.nameHashCode = name.hashCode();
    }

    /**
     * Creates a new instance of class Literal. When name = "somePredicate" and terms = {A,b,c} (where
     * A, b, c are instances of class Term) then the result is Literal somePredicate(A,b,c)
     * @param name predicate-name of the literal
     * @param terms arguments of the literal
     */
    public Literal(String name, Term ...terms){
        this(name, terms.length);
        this.set(terms);
    }

    /**
     * Creates a new instance of class Literal. When name = "somePredicate" and terms = {A,b,c} (where
     * A, b, c are instances of class Term) then the result is Literal somePredicate(A,b,c)
     * @param name predicate-name of the literal
     * @param terms arguments of the literal
     */
    public Literal(String name, List<Term> terms){
        this.id = (int)UniqueIDs.getUniqueName();
        this.terms = new Term[terms.size()];
        System.arraycopy(terms.toArray(), 0, this.terms, 0, terms.size());
        this.predicate = name.intern();
        this.nameHashCode = name.hashCode();
    } 
    
    /**
     * @param index
     * @return term in argument at position index
     */
    public Term get(int index){
        return terms[index];
    }

    /**
     * This method is used by class Clause to forbid changes of literals in it.
     * It is possible to revert this by calling this method: allowModifications(true)
     * 
     * @param allow a boolean flag indicating whether this literal can be modified or not
     */
    public void allowModifications(boolean allow){
        this.changePermitted = allow;
    }

    /**
     * This method can be used to set a term in an argument (at position index) under the condition that this literal
     * has not been locked for changes.
     * 
     * @param term term to be set
     * @param index index of the argument in which the term should be set
     */
    public void set(Term term, int index){
        if (!changePermitted)
            throw new IllegalStateException("This particular literal has been locked for changes. Create new instance and change it.");
        hashCode = -1;
        terms[index] = term;
    }

    /**
     * This method can be used to set more than one argument at once under the condition that this literal has not been
     * locked for changes.
     * 
     * @param terms array of terms which should be set as arguments, it must hold terms.length < this.arity()
     */
    public void set(Term ...terms){
        for (int i = 0; i < terms.length; i++){
            set(terms[i], i);
        }
    }

    /**
     * 
     * @return arity of the literal (i.e. number of arguments)
     */
    public int arity(){
        return terms.length;
    }
    
    /**
     * 
     * @return number of non-unique constants in the arguments
     */
    public int countConstants(){
        int count = 0;
        for (int i = 0; i < terms.length; i++){
            if (terms[i] instanceof Constant){
                count++;
            }
        }
        return count;
    }
    
    /**
     * 
     * @return predicate name of this literal
     */
    public String predicate(){
        return predicate;
    }

    /**
     * Parses Literal from its string representation, for examle:
     * Literal parsed = Literal.parse("somePredicate(X,a,f(a,b))"); It uses Prolog-convention
     * that variables start with upper-case letters.
     * 
     * @param str string representation of literal
     * @return parsed Literal
     */
    public static Literal parseLiteral(String str){
        return parseLiteral(str, fakeMapVar, fakeMapConst);
    }


    /**
     * Parses Literal from its string representation, for examle:
     * Literal parsed = Literal.parse("somePredicate(X,a,f(a,b))"); It uses Prolog-convention
     * that variables start with upper-case letters.
     * 
     * @param str string representation of literal
     * @param variables Map of used variables - it does not have to contain the variables contained in the arguments of the literal,
     * if they are not there, they will be added automatically
     * @param constants Map of used constants - it does not have to contain the constants contained in the arguments of the literal,
     * if they are not there, they will be added automatically
     * @return parsed Literal
     */
    public static Literal parseLiteral(String str, Map<String, Variable> variables, Map<String, Constant> constants){
        char c[] = str.toCharArray();
        StringBuilder predicateName = new StringBuilder();
        ArrayList<Term> arguments = new ArrayList<Term>(5);
        int index = 0;
        boolean inQuotes = false;
        boolean inDoubleQuotes = false;
        boolean ignoreNext = false;
        while (true){
            if (c[index] == '\\' && !ignoreNext){
                ignoreNext = true;
            } else {
                if (!inQuotes && !inDoubleQuotes && c[index] == '\'' && !ignoreNext){
                    predicateName.append(c[index]);
                    inQuotes = true;
                } else if (!inQuotes && !inDoubleQuotes && c[index] == '\"' && !ignoreNext){
                    predicateName.append(c[index]);
                    inDoubleQuotes = true;
                } else if (inQuotes && c[index] == '\'' && !ignoreNext){
                    predicateName.append(c[index]);
                    inQuotes = false;
                } else if (inDoubleQuotes && c[index] == '\"' && !ignoreNext){
                    predicateName.append(c[index]);
                    inDoubleQuotes = false;
                } else if (!inQuotes && !inDoubleQuotes && c[index] == '('){
                    break;
                } else {
                    predicateName.append(c[index]);
                }
                ignoreNext = false;
            }
            index++;
        }
        inQuotes = false;
        ignoreNext = false;
        while (true){
            if (index >= c.length || c[index] == ')'){
                break;
            } else if (c[index] == '(' || c[index] == ' '){
                //do nothing
            } else {
                Pair<Term,Integer> pair = ParserUtils.parseTerm(c, index, ')', variables, constants);
                arguments.add(pair.r);
                index = pair.s;
            }
            index++;
        }
        Literal retVal = new Literal(predicateName.toString().trim(), arguments.size());
        for (int i = 0; i < retVal.arity(); i++){
            retVal.set(arguments.get(i), i);
        }
        return retVal;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append(predicate).append("(");
        for (Term t : terms)
            sb.append(t).append(", ");
        if (sb.charAt(sb.length()-2) == ',')
            sb.delete(sb.length()-2, sb.length());
        sb.append(")");
        return sb.toString();
    }
    
    @Override
    public boolean equals(Object o){
        if (o == this)
            return true;
        if (!(o instanceof Literal)){
            return false;
        }
        Literal other = (Literal)o;
        if (other.terms.length != this.terms.length){
            return false;
        }
        if (!other.predicate.equals(this.predicate)){
            return false;
        }
        for (int i = 0; i < this.terms.length; i++){
            if (!terms[i].equals(other.terms[i])){
                return false;
            }
        }
        return true;
    }
    
    @Override
    public int hashCode(){
        if (hashCode != -1){
            return hashCode;
        }
        int hash = nameHashCode;
        for (int i = 0; i < terms.length; i++){
            hash = (int)((long)terms[i].hashCode()*(long)hash);
        }
        return (hashCode = hash);
    }
    
    /**
     * Every instance of class Literal has a unique id (integer). This method allows the user to get it.
     * It can be the case that l1.equals(l2) but l1.id() != l2.id()
     * 
     * @return universal identifier of the literal
     */
    public int id(){
        return id;
    }
    
    /**
     * Checks if the literal contains at least one variable
     * @return true if the literal contains at least one variable, false otherwise
     */
    public boolean containsVariable(){
        for (Term t : terms)
            if (t instanceof Variable)
                return true;
        return false;
    }
    
    /**
     * Checks if the literal contains at least one constant
     * @return true if the literal contains at least one constant, false otherwise
     */
    public boolean containsConstant(){
        for (Term t : terms)
            if (t instanceof Constant)
                return true;
        return false;
    }
    

    /**
     * 
     * @return the arguments of the literal in the form of a set
     */
    public Set<Term> terms(){
        Set<Term> set = new HashSet<Term>();
        for (Term t : terms){
            set.add(t);
        }
        return set;
    }

    /**
     * Converts Literal to Function
     * @return instance of class Function which is syntactically equivalent to this literal
     */
    public Function toFunction(){
        Function f = new Function(this.predicate, this.arity());
        for (int i = 0; i < f.arity(); i++){
            f.set(this.get(i), i);
        }
        return f;
    }

    /**
     * Creates a copy of this literal.
     * @return copy of this literal
     */
    public Literal copy(){
        Literal p = new Literal();
        p.predicate = this.predicate;
        p.terms = new Term[this.terms.length];
        p.id = this.id;
        p.hashCode = this.hashCode;
        p.nameHashCode = this.nameHashCode;
        for (int i = 0; i < terms.length; i++)
            p.terms[i] = terms[i];
        return p;
    }
}