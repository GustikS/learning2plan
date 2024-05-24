/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.ilp.basic;

import ida.utils.tuples.*;
import ida.utils.collections.FakeMap;

import java.util.*;
/**
 * Class with several useful methods for parsing.
 * @author Ondra
 */
public class ParserUtils {
    
    /**
     * Parses term which can be either variable, Constant, PrologList or Function
     * @param c array of characters in which the string representration of the term is stored
     * @param start the index at which the string representation of the term to be parsed begins
     * @param endCharacter the char which is at the end of the parsed syntactical element, e.g. ')' for literals and functions or ']' for PrologLists
     * @return pair: the first element is the parsed Term, the second element is the offset of the index to the next token.
     */
    public static Pair<Term,Integer> parseTerm(char[] c, int start, char endCharacter, Map<String,Variable> variables, Map<String,Constant> constants){
        StringBuilder term = new StringBuilder();
        int roundBrackets = 0;
        int rectBrackets = 0;
        int index = start;
        final int CONSTANT = 1, FUNCTION = 2, LIST = 3;
        int type = CONSTANT;
        boolean ignoreNext = false;
        boolean inQuotes = false;
        boolean inDoubleQuotes = false;
        Term retVal = null;
        while (index  < c.length && c[index] == ' '){
            index++;
        }
        int tStart = index;
        while (index < c.length){
            if (c[index] == '\\' && !ignoreNext){
                ignoreNext = true;
            } else {
                if (!inQuotes && !inDoubleQuotes && c[index] == '\'' && !ignoreNext){
                    term.append(c[index]);
                    inQuotes = true;
                } else if (!inQuotes && !inDoubleQuotes && c[index] == '\"' && !ignoreNext){
                    term.append(c[index]);
                    inDoubleQuotes = true;
                } else if (inQuotes && c[index] == '\'' && !ignoreNext){
                    term.append(c[index]);
                    inQuotes = false;
                } else if (inDoubleQuotes && c[index] == '\"' && !ignoreNext){
                    term.append(c[index]);
                    inDoubleQuotes = false;
                } else if (!inQuotes && !inDoubleQuotes && c[index] == '[' && !ignoreNext){
                    if (type == CONSTANT && index == tStart){
                        type = LIST;
                    }
                    term.append(c[index]);
                    rectBrackets++;
                } else if (!inQuotes && !inDoubleQuotes && c[index] == ']' && rectBrackets > 0 && !ignoreNext){
                    term.append(c[index]);
                    rectBrackets--;
                } else if (!inQuotes && !inDoubleQuotes && c[index] == '(' && !ignoreNext){
                    if (type == CONSTANT){
                        type = FUNCTION;
                    }
                    term.append(c[index]);
                    roundBrackets++;
                } else if (!inQuotes && !inDoubleQuotes && c[index] == ')' && roundBrackets > 0 && !ignoreNext){
                    term.append(c[index]);
                    roundBrackets--;
                } else if (!inQuotes && !inDoubleQuotes && roundBrackets == 0 && rectBrackets == 0 && 
                        ((c[index] == endCharacter && index == c.length-1) || c[index] == ',' || c[index] == ' ')){
                    break;
                } else {
                    term.append(c[index]);
                }
                ignoreNext = false;
            }
            index++;
        }
        String termString = term.toString().trim();
        switch (type){
            case CONSTANT:
                if (termString.length() == 0){
                    retVal = Constant.construct("");
                } else if (Character.isUpperCase(termString.charAt(0)) || termString.charAt(0) == '_'){
                    if (variables.containsKey(termString)){
                        retVal = variables.get(termString);
                    } else {
                        Variable var = Variable.construct(termString);
                        variables.put(termString, var);
                        retVal = var;
                    }
                } else {
                    if (constants.containsKey(termString)){
                        retVal = constants.get(termString);
                    } else {
                        Constant constant = Constant.construct(termString);
                        constants.put(termString, constant);
                        retVal = constant;
                    }
                }
                break;
            case LIST:
                PrologList list = PrologList.parseList(termString, variables, constants);
                retVal = list;
                break;
            case FUNCTION:
                Function f = Function.parseFunction(termString, variables, constants);
                retVal = f;
                break;
        }
        return new Pair<Term,Integer>(retVal, index);
    }
    
}