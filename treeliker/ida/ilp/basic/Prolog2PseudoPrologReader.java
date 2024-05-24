/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ida.ilp.basic;

import ida.utils.Sugar;
import ida.utils.tuples.Pair;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.List;

/**
 * Class for reading datasets represented in Prolog. This class is a sort-of a wrapper. It reads a Prolog file
 * but it behaves as if it was reading a pseudo-prolog file.
 * 
 * Individual examples in the prolog file are separated by literals:<br />
 * begin(example_id, class_label)<br />
 * ... %facts describing the example<br />
 * ... </br>
 * end(example_id)<br />
 * 
 * We assume that the Prolog file contains only ground facts.
 * 
 * Memory-intensive implementation!
 * @author Ondra
 */
public class Prolog2PseudoPrologReader extends Reader {

    private StringReader sr;
    
    /**
     * Creates a new instance of class Prolog2PseudoPrologReader.
     * @param reader Reader which should represent a Prolog file with ground facts.
     * @throws IOException
     */
    public Prolog2PseudoPrologReader(Reader reader) throws IOException {
        StringBuilder sb = new StringBuilder();
        boolean betweenBeginAndEnd = false;
        for (Pair<List<Literal>,List<Literal>> headTail : PrologParser.parse(reader)){
            Literal fact = Sugar.chooseOne(headTail.r);
            if (fact.predicate().equalsIgnoreCase("begin")){
                betweenBeginAndEnd = true;
                if (sb.length() > 0){
                    sb.append("\n");
                }
                if (fact.arity() == 1){
                    sb.append(new Literal("example_id", fact.get(0)));
                } else if (fact.arity() > 1){
                    sb.append("\"").append(LogicUtils.unquote(fact.get(1))).append("\" ").append(new Literal("example_id", fact.get(0)));
                }
            } else if (fact.predicate().equalsIgnoreCase("end")){
                betweenBeginAndEnd = false;
            } else if (betweenBeginAndEnd){
                sb.append(", ").append(fact);
            }
        }
        this.sr = new StringReader(sb.toString());
        this.sr.reset();
    }
    
    @Override
    public int read(char[] cbuf, int off, int len) throws IOException {
        return sr.read(cbuf, off, len);
    }

    @Override
    public void close() throws IOException {
        sr.close();
    }
}
