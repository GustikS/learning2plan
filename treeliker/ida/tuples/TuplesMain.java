package ida.tuples;

import ida.ilp.basic.Clause;
import ida.ilp.treeLiker.Settings;

import java.util.ArrayList;
import java.util.List;

public class TuplesMain {

    private final String[] data = {
            "_vert(1), _vert(2), _vert(3), _vert(4), bond(1, 2), bond(2, 1), bond(2, 3), bond(3, 2), bond(2, 4), bond(4, 2), bond_2(1, 2), bond_2(2, 1), bond_2(2, 3), bond_2(3, 2), bond_2(2, 4), bond_2(4, 2), red(1), blue(2), red(3), red(4)",
            "_vert(1), _vert(2), _vert(3), bond(1, 2), bond(2, 1), bond(1, 3), bond(3, 1), bond(2, 3), bond(3, 2), bond_2(1, 2), bond_2(2, 1), bond_2(1, 3), bond_2(3, 1), bond_2(2, 3), bond_2(3, 2), red(1), blue(2), green(3)"
    };

    private List<Clause> parseDataset() {
        List<Clause> clauses = new ArrayList<>();

        for (String cl : data) {
            clauses.add(Clause.parse(cl));
        }

        return clauses;
    }

    public void debug() {
        List<Clause> dataset = parseDataset();

        GraphTemplateBuilder builder = new GraphTemplateBuilder();
        builder.processExamples(dataset);
        String template = builder.inferTemplate(1);

        System.out.println(template);

        PreprocessedInput input = new PreprocessedInput(template, dataset);

        System.out.println(input);

        FeaturesTable featureData = new TreeLikerGateway().runTreeLiker(dataset, 1);
        System.out.println(featureData);
    }


    public static void main(String[] args) {
        Settings.VERBOSITY = 0;

        new TuplesMain().debug();
    }
}
