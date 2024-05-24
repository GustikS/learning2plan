/*
 * PrologReader.java
 *
 * Created on 24. leden 2008, 22:48
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package ida.ilp.basic;

import ida.utils.Sugar;
import ida.utils.tuples.*;

import java.util.*;
import java.io.*;
/**
 * Class for parsing Prolog files.
 * @author Ondra
 */
public class PrologParser {
    
    /** Creates a new instance of PrologReader */
    private PrologParser() {
    }
    
    /**
     * Parses a Prolog file into a list of rules and facts.
     * @param reader Reader of the prolog file
     * @param ignoredPredicates predicates which should be ignored
     * @return list of pairs head -: body, facts are parsed into pairs in which the first element is the fact (i.e. that literal) and the second element is null.
     * @throws IOException
     */
    public static List<Pair<List<Literal>, List<Literal>>> parse(Reader reader, String ...ignoredPredicates) throws IOException {
        final Set<String> ignoredPredicatesSet = Sugar.setFromCollections(Arrays.asList(ignoredPredicates));
        BufferedReader b = new BufferedReader(reader);
        List<Pair<List<Literal>,List<Literal>>> list = new ArrayList<Pair<List<Literal>,List<Literal>>>();
        String line = null;
        String unfinishedLine = null;
        while ((line = b.readLine()) != null){
            line = line.trim();
            if (unfinishedLine != null){
                line = unfinishedLine + line;
                unfinishedLine = null;
            }
            if (line.lastIndexOf('.') == line.length()-1 && line.lastIndexOf('.') != 0){
                try {
                    if (line.trim().length() > 0 && line.indexOf("%") == -1){
                        Pair<List<Literal>,List<Literal>> parsedLine = parseLine(line);
                        if (parsedLine.r != null){
                            parsedLine.r = Sugar.removeNulls(Sugar.funcall(parsedLine.r, new Sugar.Fun<Literal,Literal>(){
                                public Literal apply(Literal t) {
                                    if (t != null && !ignoredPredicatesSet.contains(t.predicate())){
                                        return t;
                                    } else {
                                        return null;
                                    }
                                }
                            }));
                        }
                        if (parsedLine.s != null){
                            parsedLine.s = Sugar.removeNulls(Sugar.funcall(parsedLine.s, new Sugar.Fun<Literal,Literal>(){
                                public Literal apply(Literal t) {
                                    if (t != null && !ignoredPredicatesSet.contains(t.predicate())){
                                        return t;
                                    } else {
                                        return null;
                                    }
                                }
                            }));
                        }
                        if ((parsedLine.r != null && parsedLine.r.size() > 0) || (parsedLine.s != null && parsedLine.s.size() > 0)){
                            list.add(parsedLine);
                        }
                    }
                } catch (RuntimeException rte){
                    System.out.println("Exception occured when parsing line: "+line);
                    throw rte;
                }
            } else {
                unfinishedLine = line;
            }
        }
        return list;
    }
    
    /**
     * Parses a line in Prolog format into a pair head :- body
     * @param line line to be parsed
     * @return pair head -: body, facts are parsed into pairs in which the first element is the fact (i.e. that literal) and the second element is null.
     * @throws IOException
     */
    public static Pair<List<Literal>, List<Literal>> parseLine(String line){
        if (line.indexOf("%") != -1){
            line = line.substring(0, line.indexOf("%"));
        }
        line = line.trim();
        if (line.lastIndexOf('.') == line.length()-1){
            line = line.substring(0, line.length()-1);
        }
        if (line.indexOf(":-") != -1){
            String[] split = line.split(":-");
            String head = split[0];
            String tail = split[1];
            return new Pair<List<Literal>, List<Literal>>(Sugar.listFromCollections(Clause.parse(head).literals()), Sugar.listFromCollections(Clause.parse(tail).literals()));
        } else {
            return new Pair<List<Literal>, List<Literal>>(Sugar.listFromCollections(Clause.parse(line).literals()), null);
        }
    }
    
}
