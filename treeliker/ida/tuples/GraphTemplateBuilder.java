package ida.tuples;

import ida.ilp.basic.Clause;
import ida.ilp.basic.Literal;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GraphTemplateBuilder {

    private Set<String> unaryPredicates;
    private Set<String> binaryPredicates;

    public GraphTemplateBuilder() {
        unaryPredicates = new HashSet<>();
        binaryPredicates = new HashSet<>();
    }

    public GraphTemplateBuilder(List<Clause> dataset) {
        this();
        processExamples(dataset);
    }

    public void processExamples(List<Clause> dataset) {
        unaryPredicates.clear();
        binaryPredicates.clear();

        for (Clause clause : dataset) {
            for (Literal lit : clause.literals()) {
                switch (lit.arity()) {
                    case 1:
                        unaryPredicates.add(lit.predicate());
                        break;
                    case 2:
                        binaryPredicates.add(lit.predicate());
                        break;
                    default:
                        throw new IllegalArgumentException(
                                "Cannot process literal " + lit + " of arity " + lit.arity() + "." +
                                        "Literals of arity zero or greater than two are not supported. Examples should be (labeled) graphs!");
                }
            }
        }
    }

    public String inferTemplate() {
        return inferTemplate(TuplesSettings.TEMPLATE_DEPTH);
    }

    public String inferTemplate(int depth) {
        if (depth < 0) {
            throw new IllegalArgumentException("Depth must be non-negative");
        }

        if (!unaryPredicates.contains(TuplesSettings.VERTEX_PREDICATE_NAME)) {
            throw new IllegalArgumentException("Dataset must contain the auxiliary 'VERTEX' predicate.");
        }

        StringBuilder sb = new StringBuilder();
        sb.append(TuplesSettings.VERTEX_PREDICATE_NAME);
        sb.append("(-x1)");

        for (String pred : unaryPredicates) {
            if (pred.equals(TuplesSettings.VERTEX_PREDICATE_NAME)) {
                continue;
            }

            for (int i = 1; i <= depth + 1; i++) {
                sb.append(", ");
                sb.append(pred);
                sb.append("(+").append("x").append(i).append(")");
            }
        }

        for (String pred : binaryPredicates) {
            for (int i = 1; i < depth + 1; i++) {
                sb.append(", ");
                sb.append(pred);
                sb.append("(+").append("x").append(i).append(", -").append("x").append(i + 1).append(")");
            }
        }

        return sb.toString();
    }

}
