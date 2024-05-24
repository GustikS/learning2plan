/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.ilp.basic;

import java.util.*;

import ida.utils.Sugar;
import ida.utils.tuples.Pair;
/**
 * Class for computing least general generalizations of clauses.
 * 
 * @author admin
 */
public class LGG {

    private TermLGG termLGG = new BasicTermLGG();
    
    private LiteralFilter literalFilter;
    
    /**
     * Computes least general generalization of clauses given as input. It uses Plotkin's algorithm
     * @param clauses an array of clauses
     * @return a clause which is a least general generalization of the clauses given
     * on input
     */
    public Clause lgg(Clause ...clauses){
        Clause c = clauses[0];
        for (int i = 1; i < clauses.length; i++){
            c = lgg(c, clauses[i]);
        }
        return c;
    }

    /**
     * Computes least general generalization of two clauses a and b given as inputs.
     * It uses Plotkin's algorithm.
     * 
     * @param a first clause
     * @param b second clause
     * @return a least general generalization of clauses a, b
     */
    public Clause lgg(Clause a, Clause b){
        return new Lgg().lgg(a,b);
    }

    /**
     * Sets the literal filter. Literal filter is used to filter literals which can
     * appear in the result of lgg - it allows us to use a sort of language bias.
     * @param literalFilter the literalFilter to set
     */
    public void setLiteralFilter(LiteralFilter literalFilter) {
        this.literalFilter = literalFilter;
    }
    
    private class Lgg {

        public Clause lgg(Clause a, Clause b){
            List<Literal> literals = new ArrayList<Literal>();
            TermLGG tlgg = termLGG.constructNew(Sugar.setFromCollections(a.terms(), b.terms()));
            for (Literal litA : a.literals()){
                for (Literal litB : b.literals()){
                    if (litA.predicate().equals(litB.predicate()) && litA.arity() == litB.arity()){
                        Literal c = new Literal(litA.predicate(), litA.arity());
                        for (int i = 0; i < c.arity(); i++){
                            Term ta = litA.get(i);
                            Term tb = litB.get(i);
                            if (ta instanceof Term && tb instanceof Term){
                                c.set(tlgg.termLGG(ta, tb), i);
                            }
                        }
                        if (literalFilter == null || literalFilter.filter(c)){
                            literals.add(c);
                        }
                    }
                }
            }
            return new Clause(literals);
        }
    }

    /**
     * Interface used for objects which can filter literals (e.g. according to a language bias).
     */
    public static interface LiteralFilter {
        
        /**
         * 
         * @param literal literal to be filtered
         * @return true if the literal should be included in the result of LGG,
         * false otherwise
         */
        public boolean filter(Literal literal);
        
    }
    
    /**
     * An interface that allows us to customize computation of LGGs. 
     * It defines method  termLGG(Term a, Term b) which computes LGGs of two terms.
     * It is possible to use this method to exploit taxonomical information. For example,
     * it is possible to implement the method so that termLGG(dog, mamal) = mamal
     * and not termLGG(dog, mamal) = X where X is a variable.
     */
    public static interface TermLGG {
        
        /**
         * Constructs a new instance of TermLGG class. This is called always when LGG
         * of a new pair of clauses is being computed.
         * @param termsInClauses terms in the generalized clauses
         * @return new instance of class TermLGG
         */
        public TermLGG constructNew(Set<Term> termsInClauses);
        
        /**
         * Computes LGG of terms a and b. It is guaranteed that this method will be called
         * only for computing LGG of two fixed clauses.
         * @param a
         * @param b
         * @return
         */
        public Term termLGG(Term a, Term b);
        
    }
    
    /**
     * Basic implementation of interface TermLGG - it works as specified in Plotkin's LGG
     * algorithm.
     */
    public static class BasicTermLGG implements TermLGG {
        
        private int variableIndex = 0;

        private Map<Pair<Term,Term>,Variable> usedVariables = new HashMap<Pair<Term,Term>,Variable>();

        private Set<Term> terms;
        
        /**
         * Constructs a bew instance of class BasicTermLGG
         */
        public BasicTermLGG(){}
        
        /**
         * Constructs a new instance of class BasicTermLGG.
         * @param terms the set of terms which can appear in the clauses that should be generalized.
         */
        public BasicTermLGG(Set<Term> terms){
            this.terms = terms;
        }
        
        /**
         * Computes LGG of terms as specified in the Plotkin's algorithm for computing LGGs.
         * @param a a term
         * @param b a term
         * @return a if a == b or their generalization (which is a variable) if a != b
         */
        public Term termLGG(Term a, Term b){
            if (a instanceof Function && b instanceof Function){
                return functionLGG((Function)a,(Function)b);
            } else if (a instanceof PrologList && b instanceof PrologList){
                throw new UnsupportedOperationException("Lists not supported yet");
            } else {
                return otherLGG(a,b);
            }
        }

        /**
         * Computes LGG of function symbols.
         * @param a
         * @param b
         * @return LGG of function symbols a and b as specified by Plotkin.
         */
        private Term functionLGG(Function a, Function b){
            if (a.name().equals(b.name()) && a.arity() == b.arity()){
                Function c = new Function(a.name(), a.arity());
                for (int i = 0; i < c.arity(); i++){
                    c.set(termLGG(a.get(i), b.get(i)), i);
                }
                return c;
            } else {
                return newVariable(a, b);
            }
        }

        private Term otherLGG(Term a, Term b){
            if (a instanceof Constant && b instanceof Constant && a.equals(b)){
                return a;
            } else {
                return newVariable(a, b);
            }
        }

        private Variable newVariable(Term a, Term b){
            Pair<Term,Term> pa = new Pair<Term,Term>(a,b);
            if (usedVariables.containsKey(pa)){
                return usedVariables.get(pa);
            }
            do {
                variableIndex++;
            } while (terms.contains(Variable.construct("V"+variableIndex)));
            Variable newVariable = Variable.construct("V"+variableIndex);
            usedVariables.put(new Pair<Term,Term>(a,b), newVariable);
            //usedVariables.put(new Pair<Term,Term>(b,a), newVariable);
            terms.add(newVariable);
            return newVariable;
        }
        
        public TermLGG constructNew(Set<Term> termsUsedInClauses) {
            return new BasicTermLGG(termsUsedInClauses);
        }
        
    }
    
//    public static void main(String a[]){
//        LGG lgg = new LGG();
//        System.out.println(lgg.lgg(Clause.parse("e(a,b),e(b,c),e(c,a)"),
//                Clause.parse("e(a,y),e(y,a)")));
//    }
}
