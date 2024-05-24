/*
 * Clause.java
 *
 * Created on 30. listopad 2006, 16:36
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.ilp.basic;

import ida.utils.Sugar;
import ida.utils.collections.*;
import ida.utils.tuples.*;
import java.util.*;

/**
 * Class for storing sets of positive first-order-logic literals. There is only one copy of each literal in any Clause.
 * 
 * 
 * @author Ondra
 */
public class Clause {
    
    private LinkedHashSet<Literal> literals = new LinkedHashSet<Literal>();
    
    private MultiMap<String,Literal> literalsByName;
    
    private MultiMap<Term,Literal> literalsByTerms;

    private int hashCode = -1;
    
    /**
     * Creates a new instance of class Clause. All literals in the collection "literals"
     * are locked for changes.
     * 
     * @param literals a collection of literals to be stored in this object.
     */
    public Clause(Collection<Literal> literals){
        for (Literal l : literals){
            l.allowModifications(false);
            addLiteral(l);
        }
    }

    /**
     * Adds literals from collection c.
     * @param c the collection of literals to be added.
     */
    public void addLiterals(Collection<Literal> c){
        this.hashCode = -1;
        for (Literal l : c){
            this.addLiteral(l);
        }
    }

    /**
     * Adds literal l.
     * @param literal the literal to be added.
     */
    public void addLiteral(Literal literal){
        this.hashCode = -1;
        if (!this.literals.contains(literal)){
            this.literals.add(literal);
        }
        if (this.literalsByName != null){
            this.literalsByName.put(literal.predicate(), literal);
        }
        if (this.literalsByTerms != null){
            for (int i = 0; i < literal.arity(); i++){
                this.literalsByTerms.put(literal.get(i), literal);
            }
        }
    }

    private void initLiteralsByTerms(){
        this.literalsByTerms = new MultiMap<Term,Literal>();
        for (Literal literal : literals){
            for (int i = 0; i < literal.arity(); i++){
                literalsByTerms.put(literal.get(i), literal);
            }
        }
    }
    
    private void initLiteralsByName(){
        this.literalsByName = new MultiMap<String,Literal>();
        for (Literal literal : literals){
            this.literalsByName.put(literal.predicate(), literal);
        }
    }
    
    /**
     * Checks if the set of literals contained in this clause is a subset of literals in Clause "clause".
     * @param clause the clause for which the relation "subset-of" is tested
     * @return
     */
    public boolean isSubsetOf(Clause clause){
        HashSet<Literal> set = new HashSet<Literal>();
        set.addAll(clause.literals);
        for (Literal l : literals){
            if (!set.contains(l)){
                return false;
            }
        }
        return true;
    }
    
    /**
     * 
     * @return number of unique literals
     */
    public int countLiterals(){
        return literals.size();
    }
    
    /**
     * Creates a map of frequencies of terms in the Clause "mod literals", which means
     * that it returns a Map in which terms are keys and values are the numbers of occurences
     * of these terms in literals (multiple occurence of a term in one literal is counted as one occurence). 
     * 
     * @return a map with frequencies of terms in the clause.
     */
    public Map<Term, Integer> termFrequenciesModLiterals(){
        if (literalsByTerms == null){
            initLiteralsByTerms();
        }
        HashMap<Term, Integer> frequencies = new HashMap<Term, Integer>();
        for (Map.Entry<Term,Set<Literal>> entry : this.literalsByTerms.entrySet()){
            frequencies.put(entry.getKey(), entry.getValue().size());
        }
        return frequencies;
    }

    /**
     * Creates a map of frequencies of variables in the Clause "mod literals", which means
     * that it returns a Map in which variables are keys and values are the numbers of occurences
     * of these variables in literals (multiple occurence of a variable in one literal is counted as one occurence). 
     * 
     * @return a map with frequencies of terms in the clause.
     */
    public Map<Variable, Integer> variableFrequenciesModLiterals(){
        if (literalsByTerms == null){
            initLiteralsByTerms();
        }
        HashMap<Variable, Integer> frequencies = new HashMap<Variable, Integer>();
        for (Map.Entry<Term,Set<Literal>> entry : this.literalsByTerms.entrySet()){
            if (entry.getKey() instanceof Variable)
                frequencies.put((Variable)entry.getKey(), entry.getValue().size());
        }
        return frequencies;
    }
    
    /**
     * 
     * @return literals in the Clause
     */
    public LinkedHashSet<Literal> literals(){
        return literals;
    }
    
    /**
     * 
     * Computes the set of predicate-names which appear in the clause. This method uses caching, so
     * while the first call to it may be a bit expensive, the subsequent calls should be very fast.
     * 
     * @return set of all predicate names used in the Clause
     */
    public Set<String> predicates(){
        if (literalsByName == null){
            initLiteralsByName();
        }
        return literalsByName.keySet();
    }
    
    /**
     * Collects the set of all literals in the Clause which are based on predicate name
     * "predicate". This method uses caching, so
     * while the first call to it may be a bit expensive, the subsequent calls should be very fast.
     * 
     * @param predicate the predicate name of literals that we want to obtain.
     * @return the set of literals based on predicate "predicate"
     */
    public Collection<Literal> getLiteralsByPredicate(String predicate){
        if (literalsByName == null){
            initLiteralsByName();
        }
        return literalsByName.get(predicate);
    }
    
    /**
     * Collects the set of literals which contain Term term in their arguments. This method uses caching, so
     * while the first call to it may be a bit expensive, the subsequent calls should be very fast.
     * @param term 
     * @return the set of literals which contain "term"
     */
    public Collection<Literal> getLiteralsByTerm(Term term){
        if (literalsByTerms == null){
            initLiteralsByTerms();
        }
        return literalsByTerms.get(term);
    }
    
    /**
     * Checks if the Clause contains a given literal.
     * @param literal literal whose presence is tested.
     * @return true if the Clause contains "literal", false otherwise.
     */
    public boolean containsLiteral(Literal literal){
        return this.literals.contains(literal);
    }
    
    /**
     * 
     * @return the set of all variables contained in the Clause.
     */
    public Set<Variable> variables(){
        if (literalsByTerms == null){
            initLiteralsByTerms();
        }
        HashSet<Variable> set = new HashSet<Variable>();
        for (Map.Entry<Term,Set<Literal>> entry : literalsByTerms.entrySet()){
            if (entry.getKey() instanceof Variable && entry.getValue().size() > 0){
                set.add((Variable)entry.getKey());
            }
        }
        return set;
    }
    
    /**
     * 
     * @return the set of all terms (i.e. constants, variables and function symbols) contained in the Clause.
     */
    public Set<Term> terms(){
        if (literalsByTerms == null){
            initLiteralsByTerms();
        }
        return literalsByTerms.keySet();
    }

    /**
     * Constructs a clause from its string representation. Clauses are assumed to be represented using a prolog-like syntax (which we call pseudo-prolog).
     * Variables start with upper-case letters, constants with lower-case letters, digits or apostrophes.
     * An example of a syntactically correct clause is shown below:<br />
     * <br />
     * literal(a,b), anotherLiteral(A,b), literal(b,'some longer text... ')<br />
     * 
     * @param str string representation of the Clause
     * @return new instance of the class Clause corresponding to the string representation.
     */
    public static Clause parse(String str){
        str = str.trim();
        if (str.charAt(str.length()-1) == '.'){
            str = str.substring(0, str.length()-1);
        }
        str = str+", ";
        int brackets = 0;//()
        boolean inQuotes = false;
        boolean ignoreNext = false;
        List<String> splitted = new ArrayList<String>();
        char[] chars = str.toCharArray();
        StringBuilder sb = new StringBuilder();
        int index1 = 0;
        for (char c : chars){
            if (ignoreNext){
                sb.append(c);
                ignoreNext = false;
            } else if (inQuotes) {
                sb.append(c);
                if (c == '\''){
                    inQuotes = false;
                }
            } else {
                switch (c){
                    case '\\':
                        ignoreNext = true;
                        break;
                    case '\t':
                    case ' ':
                    case '\n':
                        break;
                    case '\'':
                        inQuotes = !inQuotes;
                        sb.append(c);
                        break;
                    case '(':
                        brackets++;
                        sb.append(c);
                        break;
                    case ')':
                        brackets--;
                        sb.append(c);
                        break;
                    case ',':
                        if (brackets == 0){
                            splitted.add(sb.toString());
                            sb = new StringBuilder();
                        } else {
                            sb.append(c);
                        }
                        break;
                    default:
                        sb.append(c);
                        break;
                }
            }
            index1++;
        }
        HashMap<String,Variable> variables = new HashMap<String,Variable>();
        HashMap<String,Constant> constants = new HashMap<String,Constant>();
        List<Literal> parsedLiterals = new ArrayList<Literal>();
        for (String s : splitted){
            if (s.trim().length() > 0){
                parsedLiterals.add(Literal.parseLiteral(s.trim(), variables, constants));
            }
        }
        int anonymousIndex = 1;
        for (Literal l : parsedLiterals){
            for (int i = 0; i < l.arity(); i++){
                if (l.get(i).name().equals("_")){
                    Variable an = Variable.construct("_"+(anonymousIndex++));
                    while (variables.containsKey(an.name())){
                        an = Variable.construct("_"+(anonymousIndex++));
                    }
                    l.set(an, i);
                    variables.put(an.name(), an);
                }
            }
        }
        return new Clause(parsedLiterals);
    }
    
    @Override
    public String toString(){
        return this.toPrologLikeString();
    }
    
    private String toPrologLikeString(){
        StringBuilder sb = new StringBuilder();
        int i = 0;
        int numLiterals = this.literals.size();
        for (Literal l : this.literals){
            sb.append(l.toString());
            if (i < numLiterals-1){
                sb.append(", ");
            }
            i++;
        }
        return sb.toString();
    }
    
    @Override
    public int hashCode(){
        if (this.hashCode == -1){
            this.hashCode = this.toString().hashCode();
        }
        return this.hashCode;
    }
    
    @Override
    public boolean equals(Object o){
        if (o instanceof Clause){
            Clause c = (Clause)o;
            return this.isSubsetOf(c) && c.isSubsetOf(this);
        } else {
            return false;
        }
    }

    /**
     * Returns the set of connected components. Two parts of a clause are called term-disconnected if they have no term in common,
     * they are called variable-disconnected if they have no variable in common.
     * 
     * @return the set of term-connected components
     */
    public Collection<Clause> connectedComponents(){
        return connectedComponents(false);
    }
    
    /**
     * Returns the set of connected components. Two parts of a clause are called term-disconnected if they have no term in common,
     * they are called variable-disconnected if they have no variable in common.
     * 
     * @param justVariables
     * @return the set of term-connected components if justVariables == true, otherwise returns the set of variable-connected components
     */
    public Collection<Clause> connectedComponents(boolean justVariables){
        if (literalsByTerms == null){
            initLiteralsByTerms();
        }
        Collection<? extends Term> remainingTerms;
        if (justVariables){
            remainingTerms = this.variables();//literalsByTerms.keySet();
        } else {
            remainingTerms = literalsByTerms.keySet();
        }
        List<Clause> components = new ArrayList<Clause>();
        while (remainingTerms.size() > 0){
            Pair<Clause, Set<? extends Term>> pair = connectedComponent(Sugar.chooseOne(remainingTerms), justVariables);
            components.add(pair.r);
            remainingTerms = Sugar.collectionDifference(remainingTerms, pair.s);
        }
        if (justVariables){
            for (Literal l : this.literals){
                if (!l.containsVariable()){
                    components.add(new Clause(Sugar.set(l)));
                }
            }
        }
        return components;
    }
    
    private Pair<Clause, Set<? extends Term>> connectedComponent(Term termInComponent, boolean justVariables){
        Set<Term> closed = new HashSet<Term>();
        Stack<Term> open = new Stack<Term>();
        Set<Term> openSet = new HashSet<Term>();
        for (Literal literal : getLiteralsByTerm(termInComponent)){
            for (int i = 0; i < literal.arity(); i++){
                if (!justVariables || literal.get(i) instanceof Variable){
                    open.push(literal.get(i));
                }
            }
        }
        while (!open.isEmpty()){
            Term term = open.pop();
            if (!closed.contains(term)){
                for (Literal literal : getLiteralsByTerm(term)){
                    for (int i = 0; i < literal.arity(); i++){
                        if (!closed.contains(literal.get(i)) && !openSet.contains(literal.get(i))){
                            if (!justVariables || literal.get(i) instanceof Variable){
                                open.push(literal.get(i));
                                openSet.add(literal.get(i));
                            }
                        }
                    }
                }
                closed.add(term);
            }
        }
        Set<Literal> lits = new HashSet<Literal>();
        outerLoop: for (Literal literal : literals()){
            for (int i = 0; i < literal.arity(); i++){
                if ((!justVariables || literal.get(i) instanceof Variable) && closed.contains(literal.get(i))){
                    lits.add(literal);
                    continue outerLoop;
                }
            }
        }
        return new Pair<Clause, Set<? extends Term>>(new Clause(lits), closed);
    }
}