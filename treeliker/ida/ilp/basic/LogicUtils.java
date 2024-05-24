/*
 * LogicUtils.java
 *
 * Created on 12. leden 2008, 17:35
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.ilp.basic;

import java.util.*;

import ida.utils.*;
import ida.utils.tuples.Pair;

/**
 * Class harbouring several useful methods for manipulation with Clauses, Literals and Terms
 * @author Ondra
 */
public class LogicUtils {
    
    /** Creates a new instance of LogicUtils */
    private LogicUtils() {
    }

    /**
     * Constructs a new variable which is not contained in the given clause.
     * @param c clause that is used to constrain the possible new variables - the new variable cannot be contained in it
     * @return new variable which is not contained in Clause c
     */
    public static Variable freshVariable(Clause c){
        return freshVariable(c.variables());
    }

    /**
     * Constructs a new variable which is not contained in the given set of variables.
     * @param variables set of variables that is used to constrain the possible new variables - the new variable cannot be contained in it
     * @return
     */
    public static Variable freshVariable(Set<Variable> variables){
        int index = 0;
        Variable var = null;
        do {
            var = Variable.construct("V"+(index++));
        } while (variables.contains(var));
        return var;
    }

    /**
     * Converts a PrologList containing just instances of class Function into a Clause
     * @param pl PrologList containing just instances of class Function
     * @return clause constructed from the set of function symbols (using the method toLiteral of class Function)
     */
    public static Clause clauseFromFunctionsList(PrologList pl){
        Set<Literal> literals = new HashSet<Literal>();
        for (int i = 0; i < pl.countItems(); i++){
            Function f = (Function)pl.get(i);
            literals.add(f.toLiteral());
        }
        return new Clause(literals);
    }

    /**
     * Creates a clause from Clause c in which all occurences of Term a are replaced by Term b
     * 
     * @param c the clause
     * @param a the term to be replaced
     * @param b the term by which it should be replaced
     * @return new clause with substituted values
     */
    public static Clause substitute(Clause c, Term a, Term b){
        return substitute(c, new Term[]{a}, new Term[]{b});
    }

    /**
     * Creates a clause from Clause c in which all occurences of Terms from a are replaced by respective Terms in b
     * @param c the clause
     * @param a the terms to be replaced
     * @param b the terms by which they should be replaced
     * @return new clause with substituted values
     */
    public static Clause substitute(Clause c, Term[] a, Term[] b){
        Map<Term,Term> substitution = new HashMap<Term,Term>();
        for (int i = 0; i < a.length; i++){
            substitution.put(a[i], b[i]);
        }
        return substitute(c, substitution);
    }

    /**
     * Creates a clause from Clause c by substituting values according to a substitution represented
     * by the Map substitution. For each pair "key"-"value" in the Map, all occurences of "key" are replaced by
     * "value".
     * 
     * @param c clause on which the substitution should be applied
     * @param substitution map representing the substitution
     * @return
     */
    public static Clause substitute(Clause c, Map<Term,Term> substitution){
        Set<Literal> literals = new HashSet<Literal>();
        for (Literal l : c.literals()){
            Literal cl = l.copy();
            for (int j = 0; j < l.arity(); j++){
                if (substitution.containsKey(l.get(j))){
                    cl.set(substitution.get(l.get(j)), j);
                }
            }
            literals.add(cl);
        }
        c = new Clause(literals);
        return c;
    }

    /**
     * Creates a new literal from l in which all occurences of Terms from source are replaced by respective Terms in image
     * @param l the literal
     * @param source the terms to be replaced
     * @param image the terms by which they should be replaced
     * @return the new substituted literal
     */
    public static Literal substitute(Literal l, Term[] source, Term[] image){
        Map<Term,Term> substitution = new HashMap<Term,Term>();
        for (int i = 0; i < source.length; i++){
            substitution.put(source[i], image[i]);
        }
        return substitute(l, substitution);
    }

    /**
     * Creates a new Literal from Literal l by substituting values according to a substitution represented
     * by the Map substitution. For each pair "key"-"value" in the Map, all occurences of "key" are replaced by
     * "value".
     * 
     * @param l the literal
     * @param substitution the substitution represented as Map
     * @return the substituted literal
     */
    public static Literal substitute(Literal l, Map<Term,Term> substitution){
        Literal newLiteral = new Literal(l.predicate(), l.arity());
        for (int i = 0; i < l.arity(); i++){
            newLiteral.set(substitution.get(l.get(i)), i);
        }
        return newLiteral;
    }

    /**
     * Removes enclosing apostrophes (quotes) from a term
     * @param term the term to be unquoted
     * @return Term with removed apostrophes (quotes)
     */
    public static Term unquote(Term term){
        String name = term.name();
        if (name.length() > 0 && name.charAt(0) == '\'' && name.charAt(name.length()-1) == '\''){
            name = name.substring(1, name.length()-1);
        }
        return ParserUtils.parseTerm(name.toCharArray(), 0, ')', new HashMap<String,Variable>(), new HashMap<String,Constant>()).r;
    }

    /**
     * Creates a nice variable name for a given id. For example, for id = 0, we get
     * A, for id = 1 we get B... then A1, ..., Z1, A2,... etc.
     * 
     * @param id unique identifier of variable
     * @return string which is a name of the variable assigned to the given id
     */
    public static String niceVariableName(int id){
        if (id <= ((int)'Z'-(int)'A')){
            return String.valueOf((char)((int)'A'+id));
        } else {
            return String.valueOf((char)((int)'A'+id%((int)'Z'-(int)'A')))+(id/((int)'Z'-(int)'A'));
        }
    }

    /**
     * Creates a new clause in which it replaces all terms in a clause by the respective variables (basically it makes the
     * first letters of constants upper-case and replaces them by instances of class Variable).
     * @param c the clause
     * @return the new variabilized clause
     */
    public static Clause variabilizeClause(Clause c){
       Set<Literal> predicates = new LinkedHashSet<Literal>();
        for (Literal pred : c.literals()){
            Literal newPred = new Literal(pred.predicate(), pred.arity());
            for (int i = 0; i < pred.arity(); i++){
                newPred.set(Variable.construct(Sugar.firstCharacterToUpperCase(pred.get(i).name())), i);
            }
            predicates.add(newPred);
        }
        return new Clause(predicates);
    }

    /**
     * Creates a new clause in which it replaces all terms in a clause by the respective variables (basically it makes the
     * first letters of variables lower-case and replaces them by instances of class Constant).
     * 
     * @param c the clause
     * @return the new "constantized" clause
     */
    public static Clause constantizeClause(Clause c){
       Set<Literal> predicates = new LinkedHashSet<Literal>();
        for (Literal l : c.literals()){
            Literal newPred = new Literal(l.predicate(), l.arity());
            for (int i = 0; i < l.arity(); i++){
                newPred.set(Constant.construct(Sugar.firstCharacterToLowerCase(l.get(i).name())), i);
            }
            predicates.add(newPred);
        }
        return new Clause(predicates);
    }

    /**
     * Creates a list of predicate names which are not contained in Clause c. The list
     * will contain "count" elements.
     * @param c Clause which is used to constrain the possible predicate names - predicate already contained
     * in c cannto be contained in the generated list.
     * @param count number of predicate names to be generated
     * @return list of new predicate names
     */
    public static List<String> freshPredicateNames(Clause c, int count){
        List<String> retVal = new ArrayList<String>();
        for (int i = 0; i < Integer.MAX_VALUE; i++){
            if (retVal.size() == count){
                break;
            }
            String pred = "pred_"+i;
            if (c.getLiteralsByPredicate(pred).isEmpty()){
                retVal.add("pred"+i);
            }
        }
        return retVal;
    }

    /**
     * Checks if the given clause is ground (a clause is ground if it contains no variables)
     * @param c clause to be checked
     * @return true if c contains no variables, false otherwise
     */
    public static boolean isGround(Clause c){
        return c.variables().isEmpty();
    }

    /**
     * Given two Clauses a and b, it constructs two new equivalent clauses which
     * do not share any variables.
     * @param a the first clause
     * @param b the second clause
     * @return pair of Clauses (x,y) such that the intersection of a.variables() and b.variables() is empty
     */
    public static Pair<Clause,Clause> standardizeApart(Clause a, Clause b){
        Pair<Clause,Clause> retVal = new Pair<Clause,Clause>();
        int i = 0;
        for (Clause c : standardizeApart(Sugar.list(a,b))){
            if (i == 0){
                retVal.r = c;
            } else {
                retVal.s = c;
            }
            i++;
        }
        return retVal;
    }

    /**
     * Given two Clauses a and b, it constructs two new equivalent clauses which
     * do not share any variables.
     * @param clauses the clauses which should be standardized apart
     * @return collection of Clauses such that for any two a, b of them, the intersection of a.variables() and b.variables() is empty
     */
    public static Collection<Clause> standardizeApart(Collection<Clause> clauses){
        List<Clause> retVal = new ArrayList<Clause>();
        Map<Pair<Variable,Integer>,Variable> vars = new HashMap<Pair<Variable,Integer>,Variable>();
        Set<Variable> allVariables = new HashSet<Variable>();
        for (Clause c : clauses){
            for (Variable v : c.variables()){
                allVariables.add(v);
            }
        }
        int i = 0;
        for (Clause c : clauses){
            Set<Literal> literals = new HashSet<Literal>();
            Pair<Variable,Integer> queryPair = new Pair<Variable,Integer>();
            for (Literal l : c.literals()){
                Literal newLiteral = new Literal(l.predicate(), l.arity());
                for (int j = 0; j < l.arity(); j++){
                    if (l.get(j) instanceof Variable){
                        queryPair.set((Variable)l.get(j), i);
                        Variable var = null;
                        if ((var = vars.get(queryPair)) == null){
                            Pair<Variable,Integer> insertPair = new Pair<Variable,Integer>(queryPair.r, queryPair.s);
                            var = freshVariable(allVariables);
                            allVariables.add(var);
                            vars.put(insertPair, var);
                        }
                        newLiteral.set(var, j);
                    } else {
                        newLiteral.set(l.get(j), j);
                    }
                }
                literals.add(newLiteral);
            }
            retVal.add(new Clause(literals));
            i++;
        }
        return retVal;
    }
}
