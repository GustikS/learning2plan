/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.ilp.basic;

import ida.ilp.basic.subsumption.*;

import ida.utils.*;
import ida.utils.tuples.*;

import java.util.*;
/**
 * Class Matching encapsulates engines for computing theta-subsumption, namely ReSumEr1 and ReSumEr2 (Kuzelka, Zelezny, Fundamenta Informaticae, 2008)
 * 
 * @author ondra
 */
public class Matching {

    public final static int THETA_SUBSUMPTION = SubsumptionEngineJ2.THETA, OI_SUBSUMPTION = SubsumptionEngineJ2.OBJECT_IDENTITY, SELECTIVE_OI_SUBSUMPTION = SubsumptionEngineJ2.SELECTIVE_OBJECT_IDENTITY;

    private List<SubsumptionEngineJ2.ClauseE> positiveExamples = new ArrayList<SubsumptionEngineJ2.ClauseE>();
    
    private List<SubsumptionEngineJ2.ClauseE> negativeExamples = new ArrayList<SubsumptionEngineJ2.ClauseE>();
    
    public final static int YES = 1, NO = 2, UNDECIDED = 3;

    private int resumerVersion = 2;
    
    private SubsumptionEngineJ2 engine;

    private boolean adaptPropagationStrength = false;

    private boolean constants = true;

    private boolean learnVariableOrdering = true;

    private Random random = new Random();

    /**
     * Creates a new instance of class Matching.
     */
    public  Matching(){
        this.engine = new SubsumptionEngineJ2();
        this.engine.setRestartSequence(new IntegerFunction.Exponential(50, 2, 500));
    }

    /**
     * Creates a new instance of class Matching. It also preprocesses positive and negative examples
     * given as arguments (it computes efficient index data-structures etc).
     * 
     * @param positiveExamples list of examples (Clauses)
     * @param negativeExamples list of examples (Clauses)
     */
    public Matching(List<Clause> positiveExamples, List<Clause> negativeExamples){
        this(new SubsumptionEngineJ2(), positiveExamples, negativeExamples);
    }
    
    /**
     * Creates a new instance of class Matching. It also preprocesses positive and negative examples
     * given as arguments (it computes efficient index data-structures etc).
     * 
     * @param engine SubsumptionEngineJ2 which should be used by the Matching object.
     * @param positiveExamples list of examples (Clauses)
     * @param negativeExamples list of examples (Clauses)
     */
    public Matching(SubsumptionEngineJ2 engine, List<Clause> positiveExamples, List<Clause> negativeExamples){
        this.engine = engine;
        this.engine.setRestartSequence(new IntegerFunction.Exponential(50, 2, 500));
        for (Clause e : positiveExamples){
            if (constants){
                e = preprocessExample(e);
            }
            this.positiveExamples.add(this.engine.new ClauseE(e));
        }
        for (Clause e : negativeExamples){
            if (constants){
                e = preprocessExample(e);
            }
            this.negativeExamples.add(this.engine.new ClauseE(e));
        }
    }
    
    /**
     * Sets the sequence of "tries" in restarts of the subsumption algorithms. 
     * For example if we want to have a sequence of restarts increasing exponentially as 10 exp(index) + 100
     * then we can use new new IntegerFunction.Exponential(10, 1, 100);
     * @param f restart sequence
     */
    public void setRestartSequence(IntegerFunction f){
        this.engine.setRestartSequence(f);
    }

    /**
     * Computes coverage of Clause hypothesis on the positive examples (stored in 
     * the Matching object and set e.g. through the parameters of constructor)
     * 
     * @param hypothesis the clause for which coverage should be computed
     * @param undecided boolean array value true at position i means that theta-subsumption with
     * the corresponding positive example (at position i) should be checked. This can be useful when 
     * we have a systematic search procedure based on specialization or generalization. When it is 
     * based on specialization, then once we show that some examples cannot be covered by the given
     * hypothesis then they cannot be covered by any hypothesis obtained by specializing this hypothesis
     * therefore subsumption (which is quite costly) does not have to be computed for this particular pair hypothesis-example.
     * When searching for hypotheses in a method based on generalization, an analogical property holds (but for covered positive examples
     * )
     * 
     * @return array with results in the form of integers: Matching.YES means that the theta-subsumption holds, Matching.NO means that theta-subsumption
     * does not hold and Matching.UNDECIDED means that either we did not want to compute subsumption for the particular example or the subsumption
     * algorithm did not finish within given limit of backtracks or within given time-limit etc,
     */
    public int[] evaluatePositiveExamples(Clause hypothesis, boolean[] undecided){
        return evaluateExamples(hypothesis, positiveExamples, undecided);
    }

    /**
     * Computes coverage of Clause hypothesis on the positive examples (stored in 
     * the Matching object and set e.g. through the parameters of constructor)
     * 
     * @param hypothesis the clause for which coverage should be computed
     * @param undecided boolean array value true at position i means that theta-subsumption with
     * the corresponding positive example (at position i) should be checked. This can be useful when 
     * we have a systematic search procedure based on specialization or generalization. When it is 
     * based on specialization, then once we show that some examples cannot be covered by the given
     * hypothesis then they cannot be covered by any hypothesis obtained by specializing this hypothesis
     * therefore subsumption (which is quite costly) does not have to be computed for this particular pair hypothesis-example.
     * When searching for hypotheses in a method based on generalization, an analogical property holds (but for covered positive examples
     * )
     * 
     * @return array with results in the form of integers: Matching.YES means that the theta-subsumption holds, Matching.NO means that theta-subsumption
     * does not hold and Matching.UNDECIDED means that either we did not want to compute subsumption for the particular example or the subsumption
     * algorithm did not finish within given limit of backtracks or within given time-limit etc,
     */
    public int[] evaluateNegativeExamples(Clause hypothesis, boolean[] undecided){
        return evaluateExamples(hypothesis, negativeExamples, undecided);
    }

    private int[] evaluateExamples(Clause hypothesis, List<SubsumptionEngineJ2.ClauseE> examples, boolean[] undecided){
        int[] retVal = new int[undecided.length];
        undecided = VectorUtils.copyArray(undecided);
        Arrays.fill(retVal, UNDECIDED);
        for (Clause component : hypothesis.connectedComponents()){
            int[] result = evaluateExamplesAgainstOneComponent(component, examples, undecided);
            for (int i = 0; i < result.length; i++){
                if (undecided[i] && result[i] == NO){
                    retVal[i] = NO;
                    undecided[i] = false;
                } else if (undecided[i] && result[i] == YES){
                    retVal[i] = YES;
                }
            }
        }
        return retVal;
    }

    private int[] evaluateExamplesAgainstOneComponent(Clause hypothesis, List<SubsumptionEngineJ2.ClauseE> examples, boolean[] undecided){
        if (constants){
            hypothesis = preprocessHypothesis(hypothesis);
        }
        SubsumptionEngineJ2.ClauseStructure c = null;
        if (resumerVersion < 3){
            c = this.engine.new ClauseC(hypothesis);
        } else {
            c = this.engine.new DecomposedClauseC(hypothesis);
        }
        int[] result = new int[undecided.length];
        int j = 0;
        engine.setFirstVariableOrder(null);
        final int UP = 0, DOWN = 1;
        //int shift = Sugar.random(2);
        int shift = random.nextInt(2);
        boolean[] shiftOrNot = VectorUtils.randomBooleanVector(undecided.length, random);
        long time1 = 0;
        int maxRestart = 0;
        if (adaptPropagationStrength){
            if (shift == UP){
                engine.setArcConsistencyFrom(engine.getArcConsistencyFrom()+1);
            } else if (shift == DOWN){
                engine.setArcConsistencyFrom(Math.max(1, engine.getArcConsistencyFrom()-1));
            }
        }
        long m1 = System.currentTimeMillis();
        for (int i = 0; i < undecided.length; i++){
            if (undecided[i] && shiftOrNot[i]){
                if (j > 0 && learnVariableOrdering){
                    engine.setFirstVariableOrder(engine.getLastVariableOrder());
                }
                Boolean succ = engine.solveWithResumer(c, examples.get(i), resumerVersion);
                maxRestart = Math.max(engine.getNoOfRestarts(), maxRestart);
                if (!engine.solvedWithoutSearch()){
                    j++;
                }
                if (succ == null){
                    result[i] = UNDECIDED;
                } else if (succ.booleanValue()){
                    result[i] = YES;
                } else {
                    result[i] = NO;
                }
            }
        }
        long m2 = System.currentTimeMillis();
        time1 = m2-m1;
        if (adaptPropagationStrength){
            if (shift == UP){
                engine.setArcConsistencyFrom(engine.getArcConsistencyFrom()-1);
            } else {
                engine.setArcConsistencyFrom(Math.max(0, engine.getArcConsistencyFrom()+1));
            }
        }
        m1 = System.currentTimeMillis();
        for (int i = 0; i < undecided.length; i++){
            if (undecided[i] && !shiftOrNot[i]){
                if (j > 0 && learnVariableOrdering){
                    engine.setFirstVariableOrder(engine.getLastVariableOrder());
                }
                Boolean succ = engine.solveWithResumer(c, examples.get(i), resumerVersion);
                maxRestart = Math.max(engine.getNoOfRestarts(), maxRestart);
                if (!engine.solvedWithoutSearch()){
                    j++;
                }
                if (succ == null){
                    result[i] = UNDECIDED;
                } else if (succ.booleanValue()){
                    result[i] = YES;
                } else {
                    result[i] = NO;
                }
            }
        }
        m2 = System.currentTimeMillis();
        long time2 = m2-m1;
        double t1 = (double)time1/(double)VectorUtils.occurences(VectorUtils.and(undecided, shiftOrNot), true);
        double t2 = (double)time2/(double)VectorUtils.occurences(VectorUtils.and(undecided, VectorUtils.not(shiftOrNot)), true);
        if (t1 < t2){
            if (adaptPropagationStrength){
                if (shift == UP){
                    if (maxRestart > engine.getArcConsistencyFrom() && engine.getArcConsistencyFrom() > 1){
                        engine.setArcConsistencyFrom(engine.getArcConsistencyFrom()+1);
                    }
                } else if (shift == DOWN){
                    engine.setArcConsistencyFrom(Math.max(1, engine.getArcConsistencyFrom()-1));
                }
                //System.out.println("AC adapted: "+engine.getArcConsistencyFrom()+", "+t1+" < "+t2);
            }
        }
        return result;
    }



    /**
     * Sets the subsumption mode.
     * @param mode can be one of the following: 
     * THETA_SUBSUMPTION, OI_SUBSUMPTION, SELECTIVE_OI_SUBSUMPTION
     */
    public void setSubsumptionMode(int mode){
        this.engine.setSubsumptionMode(mode);
    }

    /**
     * Sets the first restart in which forward-checking starts to be used by the subsumption algorithm.
     * @param restart the first restart in which forward-checking should be started
     */
    public void setForwardCheckingStartsIn(int restart){
        this.engine.setForwardCheckingFrom(restart);
    }

    /**
     * Sets the first restart in which arc-consistency starts to be used by the subsumption algorithm.
     * @param restart the first restart in which arc-consistency should be started
     */
    public void setArcConsistencyStartsIn(int restart){
        this.engine.setArcConsistencyFrom(restart);
    }

    /**
     * Sets the timeout after which subsumption is considered undecided (in milliseconds)
     * @param timeout
     */
    public void setTimeout(long timeout){
        this.engine.setTimeout(timeout);
    }

    /**
     * The algorithm is able to automatically adjust the propagation strength
     * which means that it is able to automatically adjust the first restarts
     * in which forward-checking and arc-consistency start.
     * 
     * @param adaptPropagationStrength the adaptPropagationStrength to set
     */
    public void setAdaptPropagationStrength(boolean adaptPropagationStrength) {
        this.adaptPropagationStrength = adaptPropagationStrength;
    }

    /**
     * 
     * @return the subsumption engine used by the Matching object
     */
    public SubsumptionEngineJ2 getEngine(){
        return this.engine;
    }

    /**
     * The subsumption engines do not use representation of hypotheses in the form
     * of objects Clause, they represent them using objects of class SubsumptionEngineJ2.ClauseC.
     * This method allows the user to create such representations of clauses.
     * 
     * @param c the clause for which we want to create the efficient representation
     * @return the representation of Clause c as SubsumptionEngineJ2.ClauseC
     */
    public SubsumptionEngineJ2.ClauseC createClauseC(Clause c){
        return engine.createCluaseC(preprocessHypothesis(c));
    }

    /**
     * The subsumption engines do not use representation of examples in the form
     * of objects Clause, they represent them using objects of class SubsumptionEngineJ2.ClauseE.
     * This method allows the user to create such representations of examples.
     * 
     * @param e the clause for which we want to create the efficient representation
     * @return the representation of Clause e as SubsumptionEngineJ2.ClauseE
     */
    public SubsumptionEngineJ2.ClauseE createClauseE(Clause e){
        return engine.createClauseE(preprocessExample(e));
    }

    /**
     * Computes subsumption for clauses c (hypothesis), e (example).
     * 
     * @param c hypothesis
     * @param e examples
     * @return true if subsumption has been proved in the allocated time or number of backtracks,
     * false if it has been disproved in the allocated time or number of backtracks and null otherwise 
     * (i.e. if the algorithm had not been able to decide subsumption in the allocated time or number of backtracks)
     */
    public Boolean subsumption(SubsumptionEngineJ2.ClauseC c, SubsumptionEngineJ2.ClauseE e){
        return engine.solveWithResumer(c, e, resumerVersion);
    }
    
    /**
     * Computes subsumption for clauses c (hypothesis), e (example).
     * 
     * @param c hypothesis
     * @param e examples
     * @return true if subsumption has been proved in the allocated time or number of backtracks,
     * false if it has been disproved in the allocated time or number of backtracks and null otherwise 
     * (i.e. if the algorithm had not been able to decide subsumption in the allocated time or number of backtracks)
     */
    public Boolean subsumption(Clause c, Clause e){
        e = preprocessExample(e);
        for (Clause component : c.connectedComponents()){
            if (constants){
                component = preprocessHypothesis(component);
            }
            if (!engine.solveWithResumer(component, e, resumerVersion)){
                return false;
            }
        }
        return true;
    }

    /**
     * Computes all solutions of the subsumption problem "c theta-subsumes positiveExamples[exampleIndex]".
     * 
     * @param c hypothesis
     * @param exampleIndex index of the positive example for which the solutions should be computed.
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSubstitutions_P(Clause c, int exampleIndex){
        return allSubstitutions(c, exampleIndex, true);
    }

    /**
     * Computes all solutions of the subsumption problem "c theta-subsumes negativeExamples[exampleIndex]".
     * 
     * @param c hypothesis
     * @param exampleIndex index of the negative example for which the solutions should be computed.
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSubstitutions_N(Clause c, int exampleIndex){
        return allSubstitutions(c, exampleIndex, false);
    }

    private Pair<Term[],List<Term[]>> allSubstitutions(Clause c, int exampleIndex, boolean positive){
        if (constants){
            c = preprocessHypothesis(c);
        }
        SubsumptionEngineJ2.ClauseC cc = this.createClauseC(c);
        Pair<Term[],List<Term[]>> allSolutions = null;
        if (positive){
            allSolutions = engine.allSolutions(cc, this.positiveExamples.get(exampleIndex));
        } else {
            allSolutions = engine.allSolutions(cc, this.negativeExamples.get(exampleIndex));
        }
        if (!constants){
            return allSolutions;
        } else {
            boolean[] actuallyConstants = new boolean[allSolutions.r.length];
            int index = 0;
            for (Term t : allSolutions.r){
                for (Literal l : c.getLiteralsByTerm(t)){
                    if (l.predicate().indexOf("[#]") != -1){
                        actuallyConstants[index] = true;
                        break;
                    }
                    actuallyConstants[index] = false;
                }
                index++;
            }
            Term[] newTemplate = new Term[VectorUtils.occurences(actuallyConstants, false)];
            int j = 0;
            for (int i = 0; i < allSolutions.r.length; i++){
                if (!actuallyConstants[i]){
                    newTemplate[j] = allSolutions.r[i];
                    j++;
                }
            }
            List<Term[]> newSolutions = new ArrayList<Term[]>();
            for (Term[] solution : allSolutions.s){
                int k = 0;
                Term[] newSolution = new Term[newTemplate.length];
                for (int i = 0; i < solution.length; i++){
                    if (!actuallyConstants[i]){
                        newSolution[k] = solution[i];
                        k++;
                    }
                }
                newSolutions.add(newSolution);
            }
            return new Pair<Term[],List<Term[]>>(newTemplate, newSolutions);
        }
    }

    /**
     * Computes all (or at most maxCount) solutions (substitutions) of the problem "c theta-subsumes e"
     * @param c hypothesis
     * @param e example
     * @param maxCount maximum number of solutions that we want to get
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSubstitutions(Clause c, Clause e, int maxCount){
        if (constants){
            c = preprocessHypothesis(c);
            e = preprocessExample(e);
        }
        Pair<Term[],List<Term[]>> allSolutions = engine.allSolutions(c, e, maxCount);
        if (!constants){
            return allSolutions;
        } else {
            boolean[] actuallyConstants = new boolean[allSolutions.r.length];
            int index = 0;
            for (Term t : allSolutions.r){
                for (Literal l : c.getLiteralsByTerm(t)){
                    if (l.predicate().indexOf("[#]") != -1){
                        actuallyConstants[index] = true;
                        break;
                    }
                    actuallyConstants[index] = false;
                }
                index++;
            }
            Term[] newTemplate = new Term[VectorUtils.occurences(actuallyConstants, false)];
            int j = 0;
            for (int i = 0; i < allSolutions.r.length; i++){
                if (!actuallyConstants[i]){
                    newTemplate[j] = allSolutions.r[i];
                    j++;
                }
            }
            List<Term[]> newSolutions = new ArrayList<Term[]>();
            for (Term[] solution : allSolutions.s){
                int k = 0;
                Term[] newSolution = new Term[newTemplate.length];
                for (int i = 0; i < solution.length; i++){
                    if (!actuallyConstants[i]){
                        newSolution[k] = solution[i];
                        k++;
                    }
                }
                newSolutions.add(newSolution);
            }
            return new Pair<Term[],List<Term[]>>(newTemplate, newSolutions);
        }
    }

    /**
     * Computes all solutions (substitutions) of the problem "c theta-subsumes e"
     * @param c hypothesis
     * @param e example
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSubstitutions(Clause c, Clause e){
        return allSubstitutions(c, e, Integer.MAX_VALUE);
    }

    private Clause preprocessHypothesis(Clause e){
        List<Literal> literals = new ArrayList<Literal>();
        Set<Term> usedTerms = Sugar.setFromCollections(e.terms());//copy of terms
        Map<Term,Variable> selectedVariables = new HashMap<Term,Variable>();
        MutableInteger varIndex = new MutableInteger(0);
        for (Literal l : e.literals()){
            Literal copy = l.copy();
            for (int i = 0; i < l.arity(); i++){
                if (l.get(i) instanceof Constant){
                    Literal constLit = new Literal(l.get(i).name()+"[#]",1);
                    Variable variabilizedConstant = variabilize(l.get(i), varIndex, usedTerms, selectedVariables);
                    constLit.set(variabilizedConstant, 0);
                    literals.add(constLit);
                    copy.set(variabilizedConstant, i);
                } else if (l.get(i) instanceof Function){
                    
                } else if (l.get(i) instanceof PrologList){
                    
                }
            }
            literals.add(copy);
        }
        return new Clause(literals);
    }

    private Variable variabilize(Term a, MutableInteger varIndex, Set<Term> terms, Map<Term,Variable> selectedVariables){
        Variable variabilizedConstant = Variable.construct(StringUtils.capitalize(a.name()));
        if (terms.contains(variabilizedConstant)){
            if (selectedVariables.containsKey(a)){
                return selectedVariables.get(a);
            }
            do {
                varIndex.increment();
            } while (terms.contains(Variable.construct("V"+varIndex.value())));
            Variable newVariable = Variable.construct("V"+varIndex.value());
            terms.add(newVariable);
            selectedVariables.put(a, newVariable);
            return newVariable;
        } else {
            return variabilizedConstant;
        }
    }
    private Clause preprocessExample(Clause e){
        List<Literal> literals = new ArrayList<Literal>();
        for (Literal l : e.literals()){
            Literal copy = l.copy();
            for (int i = 0; i < l.arity(); i++){
                if (l.get(i) instanceof Constant){
                    Literal constLit = new Literal(l.get(i).name()+"[#]",1);
                    constLit.set(l.get(i), 0);
                    literals.add(constLit);
                    copy.set(l.get(i), i);
                } else if (l.get(i) instanceof Function){

                } else if (l.get(i) instanceof PrologList){

                }
            }
            literals.add(copy);
        }
        return new Clause(literals);
    }

    /**
     * Computes theta-reduction of Clause c.
     * @param c the close whose theta-reduction should be computed.
     * @return reduced clause
     */
    public Clause reduction(Clause c){
        if (constants){
            c = preprocessHypothesis(c);
        }
        Clause reduction = engine.reduce(c);
        if (!constants){
            return reduction;
        } else {
            Map<Term,Term> toBeChanged = new HashMap<Term,Term>();
            int index = 0;
            for (Term t : reduction.terms()){
                for (Literal l : c.getLiteralsByTerm(t)){
                    if (l.predicate().indexOf("[#]") != -1){
                        toBeChanged.put(l.get(0), Constant.construct(l.predicate().substring(0, l.predicate().indexOf("[#]"))));
                        break;
                    }
                }
                index++;
            }
            List<Literal> newLiterals = new ArrayList<Literal>();
            for (Literal l : reduction.literals()){
                Literal copy = l.copy();
                for (int i = 0; i < l.arity(); i++){
                    if (toBeChanged.containsKey(l.get(i))){
                        copy.set(toBeChanged.get(l.get(i)), i);
                    }
                }
                if (copy.predicate().indexOf("[#]") == -1){
                    newLiterals.add(copy);
                }
            }
            reduction = new Clause(newLiterals);
        }
        return reduction;
    }

    /**
     * Sets the version of ReSumEr
     * @param version 1,2,3 (3 is experimental)
     */
    public void setResumerVersion(int version){
        this.resumerVersion = version;
    }

    /**
     * The subsumption algorithms are capable to adjust the variable ordering heuristic automatically (to some extent)
     * @param learn boolean which specifies whether variable ordering heuristic should or should not beadjusted
     */
    public void setLearnVariableOrdering(boolean learn){
        this.learnVariableOrdering = learn;
    }

    /**
     * Sets the random seed for the subsumption algorithms (they are randomized).
     * @param seed seed for the random number generator
     */
    public void setRandomSeed(long seed){
        this.random = new Random(seed);
        this.engine.setRandomSeed(seed);
    }

//    public static void main(String args[]){
//        Matching m = new Matching();
//        m.setSubsumptionMode(SubsumptionEngineJ2.OBJECT_IDENTITY);
//        Clause hypo = Clause.parsePrologLikeClause("a(A,B),a(A,C)");
//        Clause e1 = Clause.parsePrologLikeClause("a(a,b),a(b,c),a(a,d)");
//        for (Term[] t : m.allSubstitutions(hypo, e1).s){
//            System.out.println(Sugar.objectArrayToString(t));
//        }
//    }
}
