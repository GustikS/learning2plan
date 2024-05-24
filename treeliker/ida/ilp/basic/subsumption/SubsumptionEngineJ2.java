/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.ilp.basic.subsumption;

import ida.ilp.basic.*;

import java.util.*;

import ida.utils.Combinatorics;
import ida.utils.random.CustomRandomGenerator;
import ida.utils.Sugar;
import ida.utils.VectorUtils;
import ida.utils.IntegerFunction;
import ida.utils.collections.ValueToIndex;
import ida.utils.collections.IntegerSet;
import ida.utils.collections.MultiMap;
import ida.utils.collections.VectorSet;
import ida.utils.tuples.*;
/**
 * This class contains implementations of RelF algorithms (Kuzelka, Zelezny, Fundamenta Informaticae 2008).
 * It is not advisable to use this class directly. It is more comfortable to use the class Matching which performs
 * preprocessing of clauses etc.
 * 
 * @author ondra
 */
public class SubsumptionEngineJ2 {

    private int lowArity = 5;

    private ValueToIndex<String> predicatesToIntegers = new ValueToIndex<String>();

    private Random random = new Random();

    private boolean learnVariableOrder = true;

    private int exploredNodesInCurrentRestart = 0;

    private int currentCutoff = Integer.MAX_VALUE;

    private int maxRestarts = Integer.MAX_VALUE;

    private int forcedVariable = -1;

    public final static int THETA = 1, OBJECT_IDENTITY = 2, SELECTIVE_OBJECT_IDENTITY = 3;

    private int subsumptionMode = THETA;

    private int forwardCheckingFrom = 1;

    private int arcConsistencyFrom = 6;

    private int[] firstVariableOrder;

    private int[] lastVariableOrder;

    private IntegerFunction restartSequence = new IntegerFunction.ConstantFunction(Integer.MAX_VALUE);

    private boolean solvedWithoutSearch = false;

    private int numberOfLastRestart = -1;

    private long timeout = Long.MAX_VALUE;

    /**
     * Computes all solutions to the subsumption problem "c theta-subsumes e"
     * @param c hypothesis
     * @param e example
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSolutions(Clause c, Clause e){
        return allSolutions(c, e, Integer.MAX_VALUE);
    }

    /**
     * Computes all solutions to the subsumption problem "c theta-subsumes e"
     * @param c hypothesis
     * @param e example
     * @param maxCount maximum number of solutions that we want to get
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSolutions(Clause c, Clause e, int maxCount){
        return allSolutions(new ClauseC(c), new ClauseE(e), maxCount);
    }

    /**
     * Computes all solutions to the subsumption problem "c theta-subsumes e"
     * @param c hypothsis
     * @param e example
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSolutions(ClauseC c, ClauseE e){
        return allSolutions(c, e, Integer.MAX_VALUE);
    }

    /**
     * Computes all solutions to the subsumption problem "c theta-subsumes e"
     * @param c hypothesis
     * @param e example 
     * @param maxCount maximum number of solutions that we want to get
     * @return pair: the first element is an array of variables, the second element is a list
     * of arrays of terms - each such array represents one solution of the subsumption problem.
     * The terms in the arrays are substitutions to the respective variables listed in the array which
     * is the first element in the pair.
     */
    public Pair<Term[],List<Term[]>> allSolutions(ClauseC c, ClauseE e, int maxCount){
        long m1 = System.currentTimeMillis();
        if (!initialUnsatCheck(c,e) || !c.initialize(e)){
            this.solvedWithoutSearch = true;
            Term[] template = new Term[c.containedIn.length];
            for (int i = 0; i < template.length; i++){
                template[i] = c.variablesToIntegers.indexToValue(i);
            }
            return new Pair<Term[],List<Term[]>>(template, new ArrayList<Term[]>(1));
        }
        List<Term[]> solutions = new ArrayList<Term[]>();
        int[] variableOrder = variableOrder(c, e, false);
        Term[] template = new Term[variableOrder.length];
        for (int i = 0; i < variableOrder.length; i++){
            template[i] = c.variablesToIntegers.indexToValue(i);
        }
        this.solvedWithoutSearch = false;
        long m2 = System.currentTimeMillis();
        solveAll(c, e, 0, variableOrder, new HashSet<Integer>(), template, solutions, maxCount);
        long m3 = System.currentTimeMillis();
//        if (solutions.size() == 0){
//            return null;
//        }
        return new Pair<Term[],List<Term[]>>(template,solutions);
    }

    private Boolean solveAll(ClauseC c, ClauseE e, int varIndex, int[] variableOrder, Set<Integer> oiSet, Term[] template, List<Term[]> solutions, int maxCount){
        if (varIndex == variableOrder.length){
            Term[] solution = new Term[variableOrder.length];
            for (int i = 0; i < variableOrder.length; i++){
                solution[i] = e.termsToIntegers.indexToValue(c.groundedValues[i]);
            }
            solutions.add(solution);
            return Boolean.TRUE;
        }
        int[] valueOrder = valueOrder(c, e, variableOrder[varIndex], 1);
        for (int i = 0; i < valueOrder.length; i++){
            if (this.subsumptionMode == OBJECT_IDENTITY && oiSet.contains(valueOrder[i])){
                continue;
            } else if (this.subsumptionMode == SELECTIVE_OBJECT_IDENTITY && !template[varIndex].name().startsWith("_") && oiSet.contains(valueOrder[i])){
                continue;
            }
            IntegerSet[] oldDomains = c.oldDomains();
            if (c.groundFC(variableOrder[varIndex], valueOrder[i], e)){
                if (solutions.size() >= maxCount){
                    return Boolean.TRUE;
                }
                if (this.subsumptionMode == OBJECT_IDENTITY){
                    oiSet.add(valueOrder[i]);
                } else if (this.subsumptionMode == SELECTIVE_OBJECT_IDENTITY && !template[varIndex].name().startsWith("_")){
                    oiSet.add(valueOrder[i]);
                }
                solveAll(c, e, varIndex+1, variableOrder, oiSet, template, solutions, maxCount);
                if (this.subsumptionMode == OBJECT_IDENTITY){
                    oiSet.remove(valueOrder[i]);
                } else if (this.subsumptionMode == SELECTIVE_OBJECT_IDENTITY && !template[varIndex].name().startsWith("_")){
                    oiSet.remove(valueOrder[i]);
                }
            }
            c.unground(variableOrder[varIndex]);
            c.restoreDomains(oldDomains);
        }
        return Boolean.FALSE;
    }

    /**
     * Solves subsumption problem "c theta-subsumes e" using Resumer1
     * @param c hypothesis
     * @param e example
     * @return Boolean.TRUE if subsumption has been proved in the given limit (time and number of backtracks),
     * false if subsumption has been disproved in the given limit and null otherwise
     */
    public Boolean solveWithResumer1(Clause c, Clause e){
        return solveWithResumer(new ClauseC(c), new ClauseE(e), 1);
    }

    /**
     * Solves subsumption problem "c theta-subsumes e" using Resumer2
     * @param c hypothesis
     * @param e example
     * @return Boolean.TRUE if subsumption has been proved in the given limit (time and number of backtracks),
     * false if subsumption has been disproved in the given limit and null otherwise
     */
    public Boolean solveWithResumer2(Clause c, Clause e){
        return solveWithResumer(new ClauseC(c), new ClauseE(e), 2);
    }

    /**
     * Solves subsumption problem "c theta-subsumes e" using Resumer3 - experimental and unpublished
     * @param c hypothesis
     * @param e example
     * @return Boolean.TRUE if subsumption has been proved in the given limit (time and number of backtracks),
     * false if subsumption has been disproved in the given limit and null otherwise
     */
    public Boolean solveWithResumer3(Clause c, Clause e){
        return solveWithResumer(new DecomposedClauseC(c), new ClauseE(e), 3);
    }

    /**
     * Solves subsumption problem "c theta-subsumes e" using Resumer1
     * @param c hypothesis
     * @param e example
     * @return Boolean.TRUE if subsumption has been proved in the given limit (time and number of backtracks),
     * false if subsumption has been disproved in the given limit and null otherwise
     */
    public Boolean solveWithResumer1(ClauseC c, ClauseE e){
        return solveWithResumer(c, e, 1);
    }

    /**
     * Solves subsumption problem "c theta-subsumes e" using Resumer2
     * @param c hypothesis
     * @param e example
     * @return Boolean.TRUE if subsumption has been proved in the given limit (time and number of backtracks),
     * Boolean.FALSE if subsumption has been disproved in the given limit and null otherwise
     */
    public Boolean solveWithResumer2(ClauseC c, ClauseE e){
        return solveWithResumer(c, e, 2);
    }

    /**
     * Solves subsumption problem "c theta-subsumes e" using specified version of Resumer
     * @param c hypothesis
     * @param e example
     * @param resumerType resumer type: 1, 2 or 3
     * @return Boolean.TRUE if subsumption has been proved in the given limit (time and number of backtracks),
     * Boolean.FALSE if subsumption has been disproved in the given limit and null otherwise
     */
    public Boolean solveWithResumer(Clause c, Clause e, int resumerType){
        if (resumerType == 3){
            return solveWithResumer(new DecomposedClauseC(c), new ClauseE(e), resumerType);
        } else {
            return solveWithResumer(new ClauseC(c), new ClauseE(e), resumerType);
        }
    }

    /**
     * 
     * @param cs hypothesis in the form of an instance ClauseStructure
     * @param e example
     * @param resumerVersion version of Resumer: 1, 2 or 3
     * @return Boolean.TRUE if subsumption has been proved in the given limit (time and number of backtracks),
     * Boolean.FALSE if subsumption has been disproved in the given limit and null otherwise
     */
    public Boolean solveWithResumer(ClauseStructure cs, ClauseE e, int resumerVersion){
        if (resumerVersion == 3){
            throw new UnsupportedOperationException();
        }
        this.numberOfLastRestart = 0;
        if (!initialUnsatCheck(cs,e)){
            this.solvedWithoutSearch = true;
            return Boolean.FALSE;
        }
        ClauseC c = null;
        if (resumerVersion <= 2){
            c = (ClauseC)cs;
        } else {
            c = ((DecomposedClauseC)cs).clauseC;
        }
        Boolean success = null;
        IntegerSet[] oldDomains = null;
        int restart = 1;
        boolean ac = false;
        long deadline = Long.MAX_VALUE;
        if (this.timeout != Long.MAX_VALUE){
            deadline = System.currentTimeMillis()+this.timeout;
        }
        if (!cs.initialize(e)){
            this.solvedWithoutSearch = true;
            this.numberOfLastRestart = restart;
            return Boolean.FALSE;
        } else if (c.literals.length == 0){
            this.solvedWithoutSearch = true;
            return Boolean.TRUE;
        }
        do {
            exploredNodesInCurrentRestart = 0;
            currentCutoff = restartSequence.f(restart)+2*c.variableDomains.length;
            int[] variableOrder;
            if (resumerVersion > 1 && restart % 2 == 0 && forcedVariable != -1){
                variableOrder = variableOrder(c, e, forcedVariable, true);
            } else {
                variableOrder = variableOrder(c, e, true);
            }
            Arrays.fill(c.groundedValues, -1);
            if (oldDomains != null){
                c.restoreDomains(oldDomains);
            }
            if (!ac && restart >= getArcConsistencyFrom()){
                if (!arcConsistencyOnProjection(c, e)){
                    this.numberOfLastRestart = restart;
                    return false;
                }
                ac = true;
                oldDomains = c.oldDomains();
            }
            Term[] template = new Term[variableOrder.length];
            if (this.subsumptionMode == SELECTIVE_OBJECT_IDENTITY){
                for (int i = 0; i < variableOrder.length; i++){
                    template[i] = c.variablesToIntegers.indexToValue(i);
                }
            }
            if (cs instanceof ClauseC){
                success = solveR(c, e, 0, variableOrder, restart, new HashSet<Integer>(), template, deadline);
            }
            //System.out.println("explored nodes: "+this.exploredNodesInCurrentRestart);
        } while (success == null && restart++ < maxRestarts && (System.currentTimeMillis() < deadline));
        this.solvedWithoutSearch = false;
        if (success == null){
            this.firstVariableOrder = null;
        }
        this.numberOfLastRestart = restart;
        return success;
    }

    /**
     * Sets the sequence of "tries" in restarts of the subsumption algorithms. 
     * For example if we want to have a sequence of restarts increasing exponentially as 10 exp(index) + 100
     * then we can use new new IntegerFunction.Exponential(10, 1, 100);
     * @param f restart sequence
     */
    public void setRestartSequence(IntegerFunction f){
        this.restartSequence = f;
    }

    /**
     * Sets the timeout after which subsumption is considered undecided (in milliseconds)
     * @param timeout
     */
    public void setTimeout(long timeout){
        this.timeout = timeout;
    }

    private boolean initialUnsatCheck(ClauseStructure c, ClauseE e){

        if (!c.predicates().isSubsetOf(e.predicates)){
            return false;
        }
        return true;
    }

    private Boolean solveR(ClauseC c, ClauseE e, int varIndex, int[] variableOrder, int restart, Set<Integer> oiSet, Term[] template, long deadline){
        if (varIndex == variableOrder.length){
            return Boolean.TRUE;
        }
        if (exploredNodesInCurrentRestart++ >= currentCutoff || (exploredNodesInCurrentRestart % 100 == 0 && System.currentTimeMillis() >= deadline)){
            return null;
        }
        int[] valueOrder = valueOrder(c, e, variableOrder[varIndex], restart);
        for (int i = 0; i < valueOrder.length; i++){
            IntegerSet[] oldDomains = c.oldDomains();
            if (this.subsumptionMode == OBJECT_IDENTITY && oiSet.contains(valueOrder[i])){
                continue;
            } else if (this.subsumptionMode == SELECTIVE_OBJECT_IDENTITY && !template[varIndex].name().startsWith("_") && oiSet.contains(valueOrder[i])){
                continue;
            }
            if ((restart < this.getForwardCheckingFrom() && c.ground(variableOrder[varIndex], valueOrder[i], e)) ||
                    (restart >= this.getForwardCheckingFrom() && c.groundFC(variableOrder[varIndex], valueOrder[i], e))){
                if (this.subsumptionMode == OBJECT_IDENTITY){
                    oiSet.add(valueOrder[i]);
                } else if (this.subsumptionMode == SELECTIVE_OBJECT_IDENTITY && !template[varIndex].name().startsWith("_")){
                    oiSet.add(valueOrder[i]);
                }
                Boolean success = solveR(c, e, varIndex+1, variableOrder, restart, oiSet, template, deadline);
                if (success == null){
                    return null;
                } else if (success.booleanValue()){
                    return true;
                }
                if (this.subsumptionMode == OBJECT_IDENTITY){
                    oiSet.remove(valueOrder[i]);
                } else if (this.subsumptionMode == SELECTIVE_OBJECT_IDENTITY && !template[varIndex].name().startsWith("_")){
                    oiSet.remove(valueOrder[i]);
                }
            } else {
                forcedVariable = variableOrder[varIndex];
            }
            c.unground(variableOrder[varIndex]);
            c.restoreDomains(oldDomains);
        }
        return Boolean.FALSE;
    }

    private int[] valueOrderForReduction(ClauseC c, ClauseE e, int variable, int restart){
        int[] valueOrder = valueOrder(c, e, variable, restart);
        String strVar = c.variablesToIntegers.indexToValue(variable).name();
        for (int i = 0; i < valueOrder.length; i++){
            String strTerm = e.termsToIntegers.indexToValue(valueOrder[i]).name();
            if (strVar.equals(strTerm)){
                VectorUtils.swap(valueOrder, 0, i);
                break;
            }
        }
        return valueOrder;
    }

    private int[] valueOrder(ClauseC c, ClauseE e, int variable, int restart){
        if (c.groundedValues[variable] == -1){
            int[] array = null;
            if (restart == 1){
                array = c.variableDomains[variable].values();
            } else {
                array = VectorUtils.copyArray(c.variableDomains[variable].values());
                VectorUtils.shuffle(array, random);
            }
            return array;
        } else {
            return new int[]{c.groundedValues[variable]};
        }
    }

    /**
     * Reduces a given clause using theta-reduction (based on Resumer)
     * @param clause the clause to be reduced
     * @return the reduced clause
     */
    public Clause reduce(Clause clause){
        Boolean success = null;
        long deadline = Long.MAX_VALUE;
        if (this.timeout != Long.MAX_VALUE){
            deadline = System.currentTimeMillis()+this.timeout;
        }
        do {
            List<Clause> clauseList = new ArrayList<Clause>();
            ClauseC c = new ClauseC(LogicUtils.variabilizeClause(clause));
            ClauseE e = new ClauseE(LogicUtils.constantizeClause(clause));
            c.initialize(e);
            int restart = 1;
            IntegerSet[] oldDomains = null;
            boolean ac = false;
            do {
                Arrays.fill(c.groundedValues, -1);
                if (oldDomains != null){
                    c.restoreDomains(oldDomains);
                }
                if (!ac && restart >= getArcConsistencyFrom()){
                    arcConsistencyOnProjection(c, e);
                    ac = true;
                    oldDomains = c.oldDomains();
                }
                int[] variableOrder = variableOrder(c, e, false);
                this.exploredNodesInCurrentRestart = 0;
                this.currentCutoff = this.restartSequence.f(restart) + 2*clause.variables().size();
                success = oneReduction(c, e, 0, variableOrder, new HashSet<Integer>(), clauseList, restart, deadline);
                //System.out.println("restart: "+restart+", explored nodes: "+this.exploredNodesInCurrentRestart);
            } while (success == null && restart++ < maxRestarts && System.currentTimeMillis() < deadline);
            if (success == Boolean.TRUE){
                clause = clauseList.get(0);
            }
        } while (success != null && success.booleanValue() && System.currentTimeMillis() < deadline);
        return LogicUtils.variabilizeClause(clause);
    }

    private Boolean oneReduction(ClauseC c, ClauseE e, int varIndex, int[] variableOrder, Set<Integer> oiSet, List<Clause> solutions, int restart, long deadline){
        if (varIndex == variableOrder.length){
            Term[] solution = new Term[variableOrder.length];
            for (int i = 0; i < variableOrder.length; i++){
                solution[i] = e.termsToIntegers.indexToValue(c.groundedValues[variableOrder[i]]);
            }
            Map<Term,Term> substitution = new HashMap<Term,Term>();
            int i = 0;
            for (int vi : variableOrder){
                Term var = c.variablesToIntegers.indexToValue(vi);
                substitution.put(var, solution[i]);
                i++;
            }
            Clause clause = LogicUtils.substitute(c.toOriginalClause(), substitution);
            if (clause.countLiterals() < c.toOriginalClause().countLiterals()){
                solutions.add(clause);
                return Boolean.TRUE;
            }
            return Boolean.FALSE;
        }
        if (exploredNodesInCurrentRestart++ >= currentCutoff ||
                (exploredNodesInCurrentRestart % 100 == 0 && System.currentTimeMillis() >= deadline)){
            return null;
        }
        int[] valueOrder = valueOrderForReduction(c, e, variableOrder[varIndex], 1);
        for (int i = 0; i < valueOrder.length; i++){
            IntegerSet[] oldDomains = c.oldDomains();
            if (c.groundFC(variableOrder[varIndex], valueOrder[i], e)){
                Boolean succ = oneReduction(c, e, varIndex+1, variableOrder, oiSet, solutions, restart, deadline);
                if (succ == null){
                    return null;
                } else if (succ.booleanValue()){
                    return Boolean.TRUE;
                }
            }
            c.unground(variableOrder[varIndex]);
            c.restoreDomains(oldDomains);
        }
        return Boolean.FALSE;
    }

    private int[] variableOrder(ClauseC c, ClauseE e, boolean ignoreSingletons){
        return variableOrder(c, e, -1, ignoreSingletons);
    }

    private int[] variableOrder(ClauseC c, ClauseE e, int fv, boolean ignoreSingletons){
        if (this.learnVariableOrder && this.firstVariableOrder != null){
            int[] ret = this.firstVariableOrder;
            this.firstVariableOrder = null;
            this.lastVariableOrder = ret;
            return ret;
        }
        int[] predicateCounts = new int[this.predicatesToIntegers.size()];
        for (int i = 0; i < e.literals.length; i += e.literals[i+1]+2){
            predicateCounts[e.literals[i]]++;
        }
        List<Integer> variableOrder = new ArrayList<Integer>();
        double[] weights = new double[c.containedIn.length];
        int index = 0;
        for (IntegerSet containedIn : c.containedIn){
            weights[index] = containedIn.size();
            weights[index] /= (double)c.variableDomains[index].size();
            index++;
        }
        double[] heuristic1 = new double[c.containedIn.length];
        if (fv == -1){
            CustomRandomGenerator crg = new CustomRandomGenerator(weights, random);
            variableOrder.add(crg.nextInt());
        } else {
            variableOrder.add(fv);
        }
        heuristic1[variableOrder.get(0)] = -1;
        for (int ci : c.containedIn[variableOrder.get(0)].values()){
            for (int i = 0; i < c.literals[ci+1]; i++){
                if (heuristic1[c.literals[ci+3+i]] != -1){
                    heuristic1[c.literals[ci+3+i]] += 1.0*weights[c.literals[ci+3+i]];
                }
            }
        }
        for (int i = 1; i < heuristic1.length; i++){
            int selected = maxIndexWithTieBreaking(heuristic1);
            heuristic1[selected] = -1;
            if (!ignoreSingletons || c.occurences[selected] > 1){
                variableOrder.add(selected);
            }
            for (int ci : c.containedIn[selected].values()){
                for (int j = 0; j < c.literals[ci+1]; j++){
                    if (heuristic1[c.literals[ci+3+j]] != -1){
                        heuristic1[c.literals[ci+3+j]] += 1.0*weights[c.literals[ci+3+j]]/predicateCounts[c.literals[ci]];
                    }
                }
            }
        }
        this.lastVariableOrder = VectorUtils.toIntegerArray(variableOrder);
        return this.lastVariableOrder;
    }

    /**
     * 
     * @return the last ordering of variables used by the algorithm 
     */
    public int[] getLastVariableOrder(){
        return this.lastVariableOrder;
    }

    /**
     * Sets the initial ordering of variables (normally, this ordering is gotten from a heuristic function),
     * @param order order of variables represented by their indices in the data structure ClauseC
     */
    public void setFirstVariableOrder(int[] order){
        this.firstVariableOrder = order;
    }

    private int maxIndexWithTieBreaking(double values[]){
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;
        int index = 0;
        int countOfEqualValues = 0;
        int[] equal = new int[values.length];
        for (double value : values){
            if (value > max){
                max = value;
                maxIndex = index;
                countOfEqualValues = 0;
            } else if (value == max){
                if (countOfEqualValues == 0){
                    equal[countOfEqualValues] = maxIndex;
                    countOfEqualValues++;
                }
                equal[countOfEqualValues] = index;
                countOfEqualValues++;
            }
            index++;
        }
        if (countOfEqualValues > 0){
            return equal[random.nextInt(countOfEqualValues)];
        }
        return maxIndex;
    }

    /**
     * Sets the subsumption mode. Aside from normal theta-subsumption, the class can work with OI-subsumption and a special version of OI subsumption.
     * @param subsumptionMode can be one of the following THETA = 1, OBJECT_IDENTITY = 2, SELECTIVE_OBJECT_IDENTITY = 3;
     */
    public void setSubsumptionMode(int subsumptionMode){
        this.subsumptionMode = subsumptionMode;
    }

    /**
     * 
     * @return true if the last solved problem has been solved without the backtracking search
     */
    public boolean solvedWithoutSearch(){
        return this.solvedWithoutSearch;
    }

    /**
     * Sets the maximum number of restarts after which the algorithm gives up and returns null instead of TRUE or FALSE.
     * @param maxRestarts the maximum number of restarts
     */
    public void setMaxRestarts(int maxRestarts) {
        this.maxRestarts = maxRestarts;
    }

    /**
     * Sets the number of first restart in which forward-checking is used.
     * @param forwardCheckingFrom the index of the first restart in which forward-checking is used
     */
    public void setForwardCheckingFrom(int forwardCheckingFrom) {
        this.forwardCheckingFrom = forwardCheckingFrom;
    }

    private boolean arcConsistencyOnProjection(ClauseC clauseC, ClauseE clauseE){
        Stack<Triple<Integer,Integer,Integer>> stack = new Stack<Triple<Integer,Integer,Integer>>();
        Map<Integer,Set<Integer>> domains = new HashMap<Integer,Set<Integer>>();
        Set<Triple<Integer,Integer,Integer>> pairs = new HashSet<Triple<Integer,Integer,Integer>>();
        for (int i = 0; i < clauseC.literals.length; i+=clauseC.literals[i+1]+3){
            if (clauseC.literals[i+1] > 1){
                for (int j = 0; j < clauseC.literals[i+1]; j++){
                    for (int k = 0; k < clauseC.literals[i+1]; k++){
                        if (clauseC.literals[i+3+j] != clauseC.literals[i+3+k]){
                            Triple<Integer,Integer,Integer> p1 = new Triple<Integer,Integer,Integer>(clauseC.literals[i+3+j], clauseC.literals[i+3+k], i);
                            if (!pairs.contains(p1)){
                                stack.push(p1);
                                pairs.add(p1);
                            }
                            if (!domains.containsKey(clauseC.literals[i+3+j])){
                                domains.put(clauseC.literals[i+3+j], clauseC.variableDomains[clauseC.literals[i+3+j]].toSet());
                            }
                        }
                    }
                }
            }
        }
        while (!stack.isEmpty()){
            Triple<Integer,Integer,Integer> triple = stack.pop();
            pairs.remove(triple);
            int oldSize = domains.get(triple.r).size();
            Set<Integer> filteredDomain = clauseC.revise(domains.get(triple.r), triple.r, domains.get(triple.s), triple.s, clauseE, triple.t);
            if (filteredDomain.size() < oldSize){
                if (filteredDomain.isEmpty()){
                    return false;
                }
                for (int neighbour : clauseC.neighbours[triple.r].values()){
                    if (neighbour != triple.r){
                        for (int neighbLit : clauseC.containedIn[neighbour].values()){
                            Triple<Integer,Integer,Integer> newTriple = new Triple<Integer,Integer,Integer>(neighbour, triple.r, neighbLit);
                            if (!pairs.contains(newTriple)){
                                stack.push(newTriple);
                                pairs.add(newTriple);
                            }
                        }
                    }
                }
                domains.put(triple.r, filteredDomain);
            }
        }
        for (Map.Entry<Integer,Set<Integer>> entry : domains.entrySet()){
            clauseC.variableDomains[entry.getKey()] = IntegerSet.createIntegerSet(entry.getValue());
        }
        return true;
    }

    /**
     * @return the arcConsistencyFrom the first restart in which arc-consistency is used
     */
    public int getArcConsistencyFrom() {
        return arcConsistencyFrom;
    }

    /**
     * Sets the number of first restart in which forward-checking is used.
     * @param arcConsistencyFrom the index of the first restart in which arc-consistency is used
     */
    public void setArcConsistencyFrom(int arcConsistencyFrom) {
        this.arcConsistencyFrom = arcConsistencyFrom;
    }

    /**
     * 
     * @return the arcConsistencyFrom the first restart in which forward-checking is used
     */
    public int getForwardCheckingFrom() {
        return forwardCheckingFrom;
    }

    /**
     * 
     * @return number of restarts used in the last run of the algorithm.
     */
    public int getNoOfRestarts(){
        return this.numberOfLastRestart-1;
    }

    /**
     * An interface implemented by data-structures for hypotheses (i.e. the clauses on
     * the left-hand-side of theta-subsumption).
     */
    public interface ClauseStructure {

        /**
         * initializes the data-structure for subsequent computation of theta-subsumption with example <em>clauseE</em>
         * @param clauseE the example with which subsumoption will be computed
         * @return true if it was not proved without search that there cannot be subsumption
         * between the hypothsis represented by this ClauseStructure object and the example <em>clauseE</em>
         */
        public boolean initialize(ClauseE clauseE);

        /**
         * 
         * @return set of integers representing predicate symbols contained in this
         * ClauseStructure. It is used for quickly refuting subsumption - when
         * predicates() is not subset of predicates contained in <em>clauseE</em>
         */
        public IntegerSet predicates();
    }

    /**
     * Decomposed clause - the decomposition is performed using GYO reduction which
     * decomposes acyclic parts of the clause (hypothesis) and quickly computes 
     * possible solutions for them - these parts then interact with the backtracking search
     */
    public class DecomposedClauseC implements ClauseStructure {

        protected ClauseC clauseC;

        private IntegerSet predicates;

        protected List<Pair<Literal,Literal>> joins = new ArrayList<Pair<Literal,Literal>>();

        private MultiMap<Integer,Pair<int[],HighArityLiterals>> specialLiterals = new MultiMap<Integer,Pair<int[],HighArityLiterals>>();

        private Set<Term> termsInCyclicCore;

        private Set<Literal> connectingLiterals;

        /**
         * Creates a new instance of class DecomposedClause - it "compiles"
         * the given Clause <em>c</em> into an efficient representation for computing
         * theta-subsumption and performs decomposition on it.
         * @param c
         */
        public DecomposedClauseC(Clause c){
            Set<Integer> preds = new HashSet<Integer>();
            for (String str : c.predicates()){
                preds.add(predicatesToIntegers.valueToIndex(str));
            }
            this.predicates = IntegerSet.createIntegerSet(preds);
            Pair<Clause,List<Pair<Literal,Literal>>> pair = SubsumptionUtils.gyoDecomposition(c);
            clauseC = new ClauseC(pair.r);
            this.termsInCyclicCore = Sugar.setFromCollections(pair.r.terms());
            joins.addAll(pair.s);
            connectingLiterals = new HashSet<Literal>();
            Set<Literal> literalsInAcyclicComponents = new HashSet<Literal>();
            for (Pair<Literal,Literal> ll : pair.s){
                literalsInAcyclicComponents.add(ll.r);
                if (ll.s != null){
                    literalsInAcyclicComponents.add(ll.s);
                }
            }
            for (Literal l : literalsInAcyclicComponents){
                for (int i = 0; i < l.arity(); i++){
                    if (termsInCyclicCore.contains(l.get(i))){
                        connectingLiterals.add(l);
                        break;
                    }
                }
            }
            System.out.println("joins: "+joins);
        }

        /**
         * 
         * @return the part of the clause that could not be decomposed using GYO reduction
         * because it was not acyclic.
         */
        public ClauseC cyclicPart(){
            return clauseC;
        }

        public boolean initialize(ClauseE example){
            specialLiterals.clear();
            if (!clauseC.initialize(example)){
                return false;
            }
            long m1 = System.currentTimeMillis();
            Map<Literal,IntegerSet> domains = acyclicArcConsistency(joins, example);
            long m2 = System.currentTimeMillis();
            if (domains == null){
                return false;
            }
            for (Literal cutLiteral : connectingLiterals){
                List<Term> commonTerms = Sugar.listFromCollections(Sugar.intersection(cutLiteral.terms(), termsInCyclicCore));
                int newArity = commonTerms.size();
                int predicate = predicatesToIntegers.valueToIndex(cutLiteral.predicate());
                int[] domain = new int[domains.get(cutLiteral).size()*(newArity+2)];
                int[] values = domains.get(cutLiteral).values();
                int[] lookup = new int[commonTerms.size()];
                for (int i = 0; i < commonTerms.size(); i++){
                    Term t = commonTerms.get(i);
                    for (int j = 0; j < cutLiteral.arity(); j++){
                        if (t == cutLiteral.get(j)){
                            lookup[i] = j;
                            break;
                        }
                    }
                }
                for (int i = 0; i < values.length; i++){
                    int val = values[i];
                    domain[i*(newArity+2)] = predicate;
                    domain[i*(newArity+2)+1] = newArity;
                    for (int j = 0; j < commonTerms.size(); j++){
                        domain[i*(newArity+2)+2+j] = example.literals[val+2+lookup[j]];
                    }
                }
                HighArityLiterals hal = new HighArityLiterals(domain, -1);
                int[] specialLiteral = new int[cutLiteral.arity()+3];
                specialLiteral[0] = predicatesToIntegers.valueToIndex(cutLiteral.predicate());
                specialLiteral[1] = newArity;
                specialLiteral[2] = -1;
                Pair<int[],HighArityLiterals> pair = new Pair<int[],HighArityLiterals>(specialLiteral, hal);
                for (int i = 0; i < newArity; i++){
                    Term t = commonTerms.get(i);
                    specialLiteral[i+3] = clauseC.variablesToIntegers.valueToIndex(t);
                    specialLiterals.put(clauseC.variablesToIntegers.valueToIndex(t), pair);
                }
            }
            long m3 = System.currentTimeMillis();
            //System.out.println((m2-m1)+" x "+(m3-m2));
            return true;
        }

        /**
         * Substitutes the value <em>value</em> for the variable at index <em>variable</em> 
         * and checks if it cannot be yet proved that there is no extension of the partial solution.
         * 
         * @param variable the variable to be grounded
         * @param value the value to be set for the variable
         * @param e the example for which subsumption is computed
         * @return true if it could not be proved that the partial solution cannot be extended
         * (whch does not mean that it can), false otherwise i.e. when it has been proved that the
         * partial solution really cannot be extended to full solution.
         */
        public boolean ground(int variable, int value, ClauseE e){
            if (!clauseC.ground(variable, value, e)){
                return false;
            }
            if (specialLiterals.size() > 0 && specialLiterals.containsKey(variable)){
                for (Pair<int[],HighArityLiterals> pair : specialLiterals.get(variable)){
                    int[] specialLiteral = pair.r;
                    HighArityLiterals hal = pair.s;
                    if (!hal.match(specialLiteral, clauseC.groundedValues, 0)){
                        return false;
                    }
                }
            }
            return true;
        }

        /**
         * Substitutes the value <em>value</em> for the variable at index <em>variable</em> 
         * and checks if it cannot be yet proved by forward-checking that there is no extension of the partial solution.
         * 
         * @param variable the variable to be grounded
         * @param value the value to be set for the variable
         * @param e the example for which subsumption is computed
         * @return true if it could not be proved by forward checking that the partial solution cannot be extended
         * (whch does not mean that it can), false otherwise i.e. when it has been proved that the
         * partial solution really cannot be extended to full solution.
         */
        protected boolean groundFC(int variable, int value, ClauseE e){
            if (!ground(variable, value, e)){
                return false;
            }
            outerLoop: for (int neighb : clauseC.neighbours[variable].values()){
                if (clauseC.groundedValues[neighb] == -1 && clauseC.containedIn[neighb].size() > 1){
                    for (int val : clauseC.variableDomains[neighb].values()){
                        boolean succ = ground(neighb, val, e);
                        clauseC.unground(neighb);
                        if (succ){
                            continue outerLoop;
                        }
                    }
                    return false;
                }
            }
            return true;
        }

        private Map<Literal,IntegerSet> acyclicArcConsistency(List<Pair<Literal,Literal>> joinTree, ClauseE example){
            Map<Literal,IntegerSet> domains = new HashMap<Literal,IntegerSet>();
            Set<Literal> literals = new LinkedHashSet<Literal>();
            for (Pair<Literal,Literal> pair : joinTree){
                literals.add(pair.r);
                if (pair.s != null){
                    literals.add(pair.s);
                }
            }
            for (Literal l : literals){
                IntegerSet rawDomain = example.domainsByPredicates[predicatesToIntegers.valueToIndex(l.predicate())];
                int domainIndex = 0;
                int[] domain = new int[rawDomain.size()];
                domainLoop: for (int eLit : rawDomain.values()){
                    for (int i = 0; i < l.arity(); i++){
                        for (int j = 0; j < l.arity(); j++){
                            if (i != j && l.get(i).equals(l.get(j))){
                                if (example.literals[eLit+2+i] != example.literals[eLit+2+j]){
                                    continue domainLoop;
                                }
                            }
                        }
                    }
                    domain[domainIndex++] = eLit;
                }
                if (domainIndex == 0){
                    return null;
                }
                domains.put(l, IntegerSet.createIntegerSet(VectorUtils.copyArray(domain, 0, domainIndex)));
            }
            for (int i = 0; i < joinTree.size(); i++){
                Pair<Literal,Literal> semijoin = joinTree.get(i);
                Literal child = semijoin.r;
                Literal parent = semijoin.s;
                if (parent != null){
                    IntegerSet domain = semijoin(parent, child, domains.get(parent), domains.get(child), example);
                    if (domain.isEmpty()){
                        return null;
                    }
                    domains.put(parent, domain);
                }
            }
            for (int i = joinTree.size()-1; i >= 0; i--){
                Pair<Literal,Literal> semijoin = joinTree.get(i);
                Literal child = semijoin.r;
                Literal parent = semijoin.s;
                if (parent != null){
                    IntegerSet domain = semijoin(child, parent, domains.get(child), domains.get(parent), example);
                    if (domain.isEmpty()){
                        return null;
                    }
                    domains.put(child, domain);
                }
            }
            return domains;
        }

        private IntegerSet semijoin(Literal literalToBeFiltered, Literal otherLiteral, IntegerSet toBeFiltered, IntegerSet other, ClauseE example){
            List<Term> commonTerms = Sugar.listFromCollections(Sugar.intersection(literalToBeFiltered.terms(), otherLiteral.terms()));
            //Set<Integer> retVal = new HashSet<Integer>();
            //MultiMap<Tuple<Integer>,Integer> mm = new MultiMap<Tuple<Integer>,Integer>();
            int[] termIndices1 = new int[commonTerms.size()];
            int[] termIndices2 = new int[commonTerms.size()];
            for (int i = 0; i < commonTerms.size(); i++){
                for (int j = 0; j < literalToBeFiltered.arity(); j++){
                    if (commonTerms.get(i).equals(literalToBeFiltered.get(j))){
                        termIndices1[i] = j;
                        break;
                    }
                }
                for (int j = 0; j < otherLiteral.arity(); j++){
                    if (commonTerms.get(i).equals(otherLiteral.get(j))){
                        termIndices2[i] = j;
                        break;
                    }
                }
            }
//            outerLoop: for (int lit : toBeFiltered.values()){
//                Tuple<Integer> tuple = new Tuple<Integer>(commonTerms.size());
//                for (int i = 0; i < termIndices1.length; i++){
//                    tuple.set(example.literals[lit+2+termIndices1[i]], i);
//                }
//                mm.put(tuple, lit);
//            }
//            LinkedHashSet<Tuple<Integer>> keyTuples = new LinkedHashSet<Tuple<Integer>>();
//            outerLoop: for (int lit : other.values()){
//                Tuple<Integer> tuple = new Tuple<Integer>(commonTerms.size());
//                for (int i = 0; i < termIndices2.length; i++){
//                    tuple.set(example.literals[lit+2+termIndices2[i]], i);
//                }
//                keyTuples.add(tuple);
//            }
//            for (Tuple<Integer> t : keyTuples){
//                retVal.addAll(Sugar.listFromCollections(mm.get(t)));
//            }
            //naive quadratic semi-join
            int retIndex = 0;
            int[] retVal = new int[toBeFiltered.size()];
            outerLoop: for (int tbfLit : toBeFiltered.values()){
                middleLoop: for (int oLit : other.values()){
                    for (int i = 0; i < termIndices1.length; i++){
                        if (example.literals[tbfLit+2+termIndices1[i]] != example.literals[oLit+2+termIndices2[i]]){
                            continue middleLoop;
                        }
                    }
                    retVal[retIndex++] = tbfLit;
                    continue outerLoop;
                }
            }
            return IntegerSet.createIntegerSet(VectorUtils.copyArray(retVal, 0, retIndex));
        }

        /**
         * 
         * @return
         */
        public IntegerSet predicates(){
            return this.predicates;
        }
    }

    /**
     * 
     */
    public class ClauseC implements ClauseStructure {

        //predicate, arity, terms
        protected int[] literals;

        private IntegerSet predicates;

        protected IntegerSet[] variableDomains;

        protected int[] groundedValues;

        private int[] occurences;

        //[term] -> literals' indices
        private IntegerSet[] containedIn;

        //[term] -> neighbours' indices in 'domains'
        private IntegerSet[] neighbours;

        private ValueToIndex<Term> variablesToIntegers = new ValueToIndex<Term>();

        /**
         * Creates a new empty ClauseC
         */
        protected ClauseC(){}

        /**
         * Creates a new instance of class ClauseC by compiling the given Clause c to 
         * an efficient representation.
         * 
         * @param c the clause on the laft-hand side of theta-subsumption (i.e. hypothesis)
         */
        public ClauseC(Clause c){
            MultiMap<Integer,Integer> integerMultiMap = new MultiMap<Integer,Integer>();
            ValueToIndex<Literal> vti = new ValueToIndex<Literal>();
            Set<Integer> predicateSet = new HashSet<Integer>();
            int literalsArrayLength = 0;
            for (Literal l : c.literals()){
                int intLiteral = vti.valueToIndex(l);
                integerMultiMap.put(predicatesToIntegers.valueToIndex(l.predicate()), intLiteral);
                literalsArrayLength += 3 + l.arity();
                predicateSet.add(predicatesToIntegers.valueToIndex(l.predicate()));
            }
            this.predicates = IntegerSet.createIntegerSet(predicateSet);
            literals = new int[literalsArrayLength];
            Map<Literal,Integer> intLitMap = new HashMap<Literal,Integer>();
            Map<Integer,Literal> litIntMap = new HashMap<Integer,Literal>();
            int index = 0;
            int literalIndex = 0;
            for (Literal l : c.literals()){
                literals[index] = predicatesToIntegers.valueToIndex(l.predicate());
                literals[index+1] = l.arity();
                literals[index+2] = literalIndex;
                intLitMap.put(l, index);
                litIntMap.put(index, l);
                index+=3;
                for (int j = 0; j < l.arity(); j++){
                    literals[index+j] = variablesToIntegers.valueToIndex(l.get(j));
                }
                index += l.arity();
                literalIndex++;
            }
            containedIn = new IntegerSet[variablesToIntegers.size()];
            occurences = new int[variablesToIntegers.size()];
            MultiMap<Integer,Integer> containedInBag = new MultiMap<Integer,Integer>();
            for (Literal l : c.literals()){
                for (int i = 0; i < l.arity(); i++){
                    containedInBag.put(variablesToIntegers.valueToIndex(l.get(i)), intLitMap.get(l));
                    occurences[variablesToIntegers.valueToIndex(l.get(i))]++;
                }
            }
            for (Map.Entry<Integer,Set<Integer>> entry : containedInBag.entrySet()){
                int term = entry.getKey();
                containedIn[term] = IntegerSet.createIntegerSet(entry.getValue());
            }
            neighbours = new IntegerSet[containedIn.length];
            for (int i = 0; i < containedIn.length; i++){
                Set<Integer> set = new HashSet<Integer>();
                for (int literalsIndex : containedIn[i].values()){
                    for (int j = literalsIndex+3; j < literalsIndex+3+literals[literalsIndex+1]; j++){
                        if (literals[j] != i){
                            set.add(literals[j]);
                        }
                    }
                }
                neighbours[i] = IntegerSet.createIntegerSet(set);
            }
            variableDomains = new IntegerSet[c.variables().size()];
            groundedValues = new int[c.variables().size()];
            Arrays.fill(groundedValues, -1);
        }

        public boolean initialize(ClauseE e){
            Arrays.fill(this.variableDomains, null);
            for (int i = 0; i < this.variableDomains.length; i++){
                for (int ciLit : this.containedIn[i].values()){
                    for (int j = 0; j < literals[ciLit+1]; j++){
                        if (this.literals[ciLit+j+3] == i){
                            if (this.variableDomains[i] == null){
                                this.variableDomains[i] = e.variableDomains.get(new Pair<Integer,Integer>(this.literals[ciLit], j));
                            } else {
                                this.variableDomains[i] = IntegerSet.intersection(e.variableDomains.get(new Pair<Integer,Integer>(this.literals[ciLit], j)), this.variableDomains[i]);
                            }
                        }
                    }
                }
                if (this.variableDomains[i].isEmpty()){
                    return false;
                }
            }
            long m2 = System.currentTimeMillis();
            Arrays.fill(groundedValues, -1);
            return true;
        }

        /**
         * Substitutes the value <em>value</em> for the variable at index <em>variable</em> 
         * and checks if it cannot be yet proved that there is no extension of the partial solution.
         * 
         * @param variable the variable to be grounded
         * @param value the value to be set for the variable
         * @param e the example for which subsumption is computed
         * @return true if it could not be proved that the partial solution cannot be extended
         * (whch does not mean that it can), false otherwise i.e. when it has been proved that the
         * partial solution really cannot be extended to full solution.
         */
        protected boolean ground(int variable, int value, ClauseE e){
            this.groundedValues[variable] = value;
            for (int ciLit : containedIn[variable].values()){
                if (!e.checkLiteral(literals, groundedValues, ciLit)){
                    return false;
                }
            }
            return true;
        }

        /**
         * Substitutes the value <em>value</em> for the variable at index <em>variable</em> 
         * and checks if it cannot be yet proved using forward checking that there is no extension of the partial solution.
         * 
         * @param variable the variable to be grounded
         * @param value the value to be set for the variable
         * @param e the example for which subsumption is computed
         * @return true if it could not be proved using forward checking that the partial solution cannot be extended
         * (whch does not mean that it can), false otherwise i.e. when it has been proved that the
         * partial solution really cannot be extended to full solution.
         */
        protected boolean groundFC(int variable, int value, ClauseE e){
            if (!ground(variable, value, e)){
                return false;
            }
            outerLoop: for (int neighb : this.neighbours[variable].values()){
                if (this.groundedValues[neighb] == -1 && this.containedIn[neighb].size() > 1){
                    for (int val : this.variableDomains[neighb].values()){
                        boolean succ = ground(neighb, val, e);
                        unground(neighb);
                        if (succ){
                            continue outerLoop;
                        }
                    }
                    return false;
                }
            }
            return true;
        }

        //revise function of AC-3 algorithm (arc consistency)
        private Set<Integer> revise(Set<Integer> domain1, int var1, Set<Integer> domain2, int var2, ClauseE e, int literal){
            Set<Integer> filtered = new LinkedHashSet<Integer>();
            if (groundedValues[var1] == -1 && groundedValues[var2] == -1){
                for (Integer d1 : domain1){
                    this.groundedValues[var1] = d1;
                    for (Integer d2 : domain2){
                        this.groundedValues[var2] = d2;
                        if (e.checkLiteral(literals, groundedValues, literal)){
                            filtered.add(d1);
                            unground(var2);
                            break;
                        }
                        unground(var2);
                    }
                    unground(var1);
                }
            } else if (groundedValues[var1] > -1){
                filtered.add(groundedValues[var1]);
            } else if (groundedValues[var1] == -1 && groundedValues[var2] > -1){
                for (Integer d1 : domain1){
                    this.groundedValues[var1] = d1;
                    if (e.checkLiteral(literals, groundedValues, literal)){
                        filtered.add(d1);
                    }
                    unground(var1);
                }
            } else {
                return domain1;
            }
            return filtered;
        }

        /**
         * Restores domains of variables to values contained in <em>oldDomains</em>
         * @param oldDomains the values that should be restored
         */
        protected void restoreDomains(IntegerSet[] oldDomains){
            this.variableDomains = oldDomains;
        }

        /**
         * 
         * @return creates a copy of domains of this ClauseC, these can later be used 
         * in restoreDomains(...)
         */
        protected IntegerSet[] oldDomains(){
            IntegerSet[] oldDoms = new IntegerSet[this.variableDomains.length];
            System.arraycopy(this.variableDomains, 0, oldDoms, 0, oldDoms.length);
            return oldDoms;
        }

        /**
         * Ungrounds variable <em>variable</em>.
         * @param variable the variable to be unground
         */
        protected void unground(int variable){
            this.groundedValues[variable] = -1;
        }

        @Override
        public String toString(){
            return toClause().toString();
        }

        /**
         * 
         * @return original clause represented by this ClauseC object
         */
        public Clause toOriginalClause(){
            List<Literal> lits = new ArrayList<Literal>();
            for (int i = 0; i < literals.length; i+=literals[i+1]+3){
                Literal l = new Literal(predicatesToIntegers.indexToValue(literals[i]), literals[i+1]);
                for (int j = 0; j < literals[i+1]; j++){
                    l.set(variablesToIntegers.indexToValue(literals[i+3+j]), j);
                }
                lits.add(l);
            }
            return new Clause(lits);
        }

        /**
         * 
         * @return representation of this ClauseC object as an instance of class Clause
         */
        public Clause toClause(){
            List<Literal> lits = new ArrayList<Literal>();
            for (int i = 0; i < literals.length; i+=literals[i+1]+3){
                Literal l = new Literal(predicatesToIntegers.indexToValue(literals[i]), literals[i+1]);
                for (int j = 0; j < literals[i+1]; j++){
                    if (groundedValues[literals[i+3+j]] != -1){
                        l.set(Constant.construct(String.valueOf(groundedValues[literals[i+3+j]])), j);
                    } else {
                        l.set(variablesToIntegers.indexToValue(literals[i+3+j]), j);
                    }
                }
                lits.add(l);
            }
            return new Clause(lits);
        }

        /**
         * 
         * @return the set of predicates (represented as integers) contained in this ClauseC
         */
        public IntegerSet predicates(){
            return this.predicates;
        }

        /**
         * [template,assignment]
         * @param example the example for which subsumption is computed
         * @return  Pair:  the first element is an array of variables, the second element is a an
         * array of terms - represents the groundings of the variables.
         * The terms in the second array are substitutions to the respective variables listed in the array which
         * is the first element in the pair.
         */
        public Pair<Term[],Term[]> getVariableAssignment(ClauseE example){
            Term[] template = new Term[this.groundedValues.length];
            Term[] assignment = new Term[this.groundedValues.length];
            for (int i = 0; i < template.length; i++){
                template[i] = this.variablesToIntegers.indexToValue(i);
                assignment[i] = example.termsToIntegers.indexToValue(this.groundedValues[i]);
            }
            return new Pair<Term[],Term[]>(template, assignment);
        }
    }

    /**
     * 
     */
    public class ClauseE {

        protected ValueToIndex<Term> termsToIntegers = new ValueToIndex<Term>();

        protected Map<Pair<Integer,Integer>,IntegerSet> variableDomains = new HashMap<Pair<Integer,Integer>,IntegerSet>();

        protected int[] literals;

        private IntegerSet predicates;

        protected IntegerSet[] domainsByPredicates;

        private LowArityLiterals lal;

        private HighArityLiterals hal;

        /**
         * Creates a new instance of class ClauseE which serves as an efficient data-structure
         * for storing the clauses that are on the right-hand side of theta-subsumption relation (i.e. the examples).
         * 
         * @param c the clause which should be compiled into the efficient representation
         */
        public ClauseE(Clause c){
            MultiMap<Integer,Integer> integerMultiMap = new MultiMap<Integer,Integer>();
            Set<Integer> predicates = new HashSet<Integer>();
            int literalIndex = 0;
            for (Literal l : c.literals()){
                integerMultiMap.put(predicatesToIntegers.valueToIndex(l.predicate()), literalIndex);
                literalIndex += 2 + l.arity();
                predicates.add(predicatesToIntegers.valueToIndex(l.predicate()));
            }
            this.predicates = IntegerSet.createIntegerSet(predicates);
            domainsByPredicates = new IntegerSet[predicatesToIntegers.size()];
            for (Map.Entry<Integer,Set<Integer>> entry : integerMultiMap.entrySet()){
                domainsByPredicates[entry.getKey()] = IntegerSet.createIntegerSet(entry.getValue());
            }
            literals = new int[literalIndex];
            Map<Literal,Integer> intLitMap = new HashMap<Literal,Integer>();
            Map<Integer,Literal> litIntMap = new HashMap<Integer,Literal>();
            int index = 0;
            MultiMap<Pair<Integer,Integer>,Integer> varDomains = new MultiMap<Pair<Integer,Integer>,Integer>();
            for (Literal l : c.literals()){
                literals[index] = predicatesToIntegers.valueToIndex(l.predicate());
                literals[index+1] = l.arity();
                intLitMap.put(l, index);
                litIntMap.put(index, l);
                index+=2;
                for (int j = 0; j < l.arity(); j++){
                    literals[index+j] = termsToIntegers.valueToIndex(l.get(j));
                    varDomains.put(new Pair<Integer,Integer>(literals[index-2], j), literals[index+j]);
                }
                index += l.arity();
            }
            for (Map.Entry<Pair<Integer,Integer>,Set<Integer>> entry : varDomains.entrySet()){
                this.variableDomains.put(entry.getKey(), IntegerSet.createIntegerSet(entry.getValue()));
            }
            lal = new LowArityLiterals(literals, lowArity);
            hal = new HighArityLiterals(literals, lowArity);
        }

        /**
         * Checks if the literal at position <em>index</em> in the array <em>cliterals</em> (which is taken
         * from ClauseC) can be extended to a literal which is contained in this ClauseE.
         * @param cliterals representation of literals from ClauseC
         * @param grounding grounding of variables in the ClauseC (these are referenced from the cliterals array)
         * @param index index at which the checked literal is located in the array cliterals
         * @return true of the literal can be extended so that it would be equal to some literal from this ClauseE, false otherwise
         */
        public boolean checkLiteral(int[] cliterals, int[] grounding, int index){
            if (cliterals[index+1] <= lowArity){
                return this.lal.match(cliterals, grounding, index);
            } else {
                return this.hal.match(cliterals, grounding, index);
            }
        }

        /**
         * 
         * @param literal the integer representation of the literal for which we want the string representation
         * @return string representation of literal represented by integer <em>literal</em>
         */
        public String literalToString(int literal){
            StringBuilder sb = new StringBuilder();
            sb.append(predicatesToIntegers.indexToValue(literals[literal]));
            sb.append("(");
            for (int i = 0; i < literals[literal+1]; i++){
                sb.append(termsToIntegers.indexToValue(literals[literal+2+i]));
                if (i < literals[literal+1]-1){
                    sb.append(", ");
                }
            }
            sb.append(")");
            return sb.toString();
        }
    }

    private static class HighArityLiterals {

        private Map<Triple<Integer,Integer,Integer>,Integer> lower;

        private Map<Triple<Integer,Integer,Integer>,Integer> upper;

        private int[] literals;

        private int maxArity;

        /**
         * 
         * @param lits
         * @param maxArity
         */
        public HighArityLiterals(int[] lits, int maxArity){
            this.maxArity = maxArity;
            List<Integer> tempLiterals = new ArrayList<Integer>();
            //[predicate,argument,term] -> position in literals array
            MultiMap<Triple<Integer,Integer,Integer>,Integer> bag = new MultiMap<Triple<Integer,Integer,Integer>,Integer>();
            for (int i = 0; i < lits.length; i += lits[i+1]+2){
                if (lits[i+1] > this.maxArity){
                    for (int j = 0; j < lits[i+1]+2; j++){
                        tempLiterals.add(lits[i+j]);
                    }
                }
            }
            this.literals = VectorUtils.toIntegerArray(tempLiterals);
            for (int i = 0; i < this.literals.length; i += this.literals[i+1]+2){
                for (int j = 0; j < this.literals[i+1]; j++){
                    bag.put(new Triple<Integer,Integer,Integer>(this.literals[i], j, this.literals[i+2+j]), i);
                }
            }
            this.lower = new HashMap<Triple<Integer,Integer,Integer>,Integer>();
            this.upper = new HashMap<Triple<Integer,Integer,Integer>,Integer>();
            for (Map.Entry<Triple<Integer,Integer,Integer>,Set<Integer>> entry : bag.entrySet()){
                this.lower.put(entry.getKey(), Sugar.findBest(entry.getValue(), new Sugar.MyComparator<Integer>() {
                    public boolean isABetterThanB(Integer a, Integer b) {
                        return a < b;
                    }
                }));
                this.upper.put(entry.getKey(), Sugar.findBest(entry.getValue(), new Sugar.MyComparator<Integer>() {
                    public boolean isABetterThanB(Integer a, Integer b) {
                        return a >= b;
                    }
                }));
            }
        }

        /**
         * 
         * @param cliterals
         * @param grounding
         * @param index
         * @return
         */
        public boolean match(int[] cliterals, int[] grounding, int index){
            int lowerBound = 0;
            int upperBound = this.literals.length;
            int predicate = cliterals[index];
            int[] cliteral = new int[cliterals[index+1]+2];
            cliteral[0] = cliterals[index];
            cliteral[1] = cliterals[index+1];
            Triple<Integer,Integer,Integer> t = new Triple<Integer,Integer,Integer>(predicate, 0, 0);
            for (int i = index+3, j = 0; i < index+cliterals[index+1]+3; i++, j++){
                if (grounding[cliterals[i]] == -1){
                    cliteral[j+2] = -maxArity-2;
                } else {
                    cliteral[j+2] = grounding[cliterals[i]];
                    t.s = j;
                    t.t = grounding[cliterals[i]];
                    Integer fromLower, fromUpper;
                    if ((fromLower = this.lower.get(t)) == null || (fromUpper = this.upper.get(t)) == null){
                        return false;
                    }
                    lowerBound = Math.max(lowerBound, fromLower);
                    upperBound = Math.min(upperBound, fromUpper);
                }
            }
            int iters = 0;
            outerLoop: for (int i = lowerBound; i <= upperBound; i += this.literals[i+1]+2){
                iters++;
                for (int j = 0; j < cliteral.length; j++){
                    if (cliteral[j] > -1 && this.literals[i+j] != cliteral[j]){
                        continue outerLoop;
                    }
                }
                return true;
            }
            return false;
        }
    }

    /**
     * 
     */
    private static class LowArityLiterals {

        private VectorSet set = new VectorSet();

        private int maxArity;

        /**
         * 
         * @param literals
         * @param indices
         * @param maxArity
         */
        public LowArityLiterals(int[] literals, int[] indices, int maxArity){
            this.maxArity = maxArity;
            for (int i : indices){
                add(literals, i);
            }
        }

        /**
         * 
         * @param literals
         * @param maxArity
         */
        public LowArityLiterals(int[] literals, int maxArity){
            this.maxArity = maxArity;
            for (int i = 0; i < literals.length; i += literals[i+1]+2){
                add(literals, i);
            }
        }

        /**
         * 
         * @param literals
         * @param index
         */
        public void add(int[] literals, int index){
            int arity = literals[index+1];
            if (arity <= maxArity){
                List<Integer> list = new ArrayList<Integer>();
                for (int j = 0; j < arity; j++){
                    list.add(j);
                }
                List<Tuple<Integer>> subsequences = Combinatorics.allSubsequences(list);
                for (Tuple<Integer> t : subsequences){
                    int[] literal = new int[arity+2];
                    Arrays.fill(literal, -maxArity-2);
                    literal[0] = literals[index];
                    literal[1] = literals[index+1];
                    for (int k = 0; k < t.size(); k++){
                        literal[t.get(k)+2] = literals[index+2+t.get(k)];
                    }
                    set.add(literal);
                }
            }
        }

        /**
         * 
         * @param cliterals
         * @param grounding
         * @param index
         * @return
         */
        public boolean match(int[] cliterals, int[] grounding, int index){
            int[] cliteral = new int[cliterals[index+1]+2];
            cliteral[0] = cliterals[index];
            cliteral[1] = cliterals[index+1];
            for (int i = index+3, j = 0; i < index+cliterals[index+1]+3; i++, j++){
                if (grounding[cliterals[i]] == -1){
                    cliteral[j+2] = -maxArity-2;
                } else {
                    cliteral[j+2] = grounding[cliterals[i]];
                }
            }
            //System.out.println(VectorUtils.intArrayToString(cliteral)+" "+set.contains(cliteral));
            return set.contains(cliteral);
        }
    }

    /**
     * 
     * @param c
     * @return
     */
    public ClauseC createCluaseC(Clause c){
        return new ClauseC(c);
    }

    /**
     * 
     * @param e
     * @return
     */
    public ClauseE createClauseE(Clause e){
        return new ClauseE(e);
    }

    /**
     * 
     * @param seed
     */
    public void setRandomSeed(long seed){
        this.random = new Random(seed);
    }

    /**
     * 
     * @param lowArity
     */
    public void setWhatIsLowArity(int lowArity){
        this.lowArity = lowArity;
    }

//    public static void main(String args[]){
//        Clause c = Clause.parsePrologLikeClause("a(A,B), b(B,C), c(C,D), d(D,E), e(E,F), f(F,D)");
//        Clause e = Clause.parsePrologLikeClause("a(a,b), a(b,a), b(b,c), c(c,d), d(d,e), e(e,f), f(f,d), f(d,e)");
//        SubsumptionEngineJ2 sej2 = new SubsumptionEngineJ2();
//        DecomposedClauseC dcc = sej2.new DecomposedClauseC(c);
//        ClauseE ce = sej2.new ClauseE(e);
//        dcc.initialize(ce);
//
//
//    }
}
