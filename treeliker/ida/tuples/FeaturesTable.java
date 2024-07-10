package ida.tuples;

import ida.ilp.basic.Clause;
import ida.ilp.basic.Literal;
import ida.ilp.treeLiker.Block;
import ida.ilp.treeLiker.Dataset;
import ida.ilp.treeLiker.Example;
import ida.ilp.treeLiker.Table;

import java.util.*;

public class FeaturesTable {

    public final List<Clause> features;
    public final int[][] table;



    public static FeaturesTable fromTreeLikerTable(Table<Integer, String> propTable) {
        return new FeaturesTable(propTable);
    }

    public static FeaturesTable forGraphExplanations(Set<Block> features, Dataset dataset) {
        return new FeaturesTable(features, dataset);
    }



    private FeaturesTable(Table<Integer, String> propTable) {
        Set<String> stringFeatures = propTable.filteredAttributes();
        int noExamples = propTable.getAttributeVector(stringFeatures.iterator().next()).size();

        features = new ArrayList<>(stringFeatures.size());
        table = new int[noExamples][stringFeatures.size()];

        int featureIndex = 0;
        for (String strFeature : stringFeatures) {
            Clause feature = Clause.parse(strFeature);

            features.add(feature);

            Map<Integer, String> values = propTable.getAttributeVector(strFeature);
            for (Map.Entry<Integer, String> pair : values.entrySet()) {
                table[pair.getKey()][featureIndex] = Integer.parseInt(pair.getValue());
            }

            featureIndex++;
        }

    }

    private FeaturesTable(Set<Block> blocks, Dataset dataset) {
        this.features = new ArrayList<>(blocks.size());

        for (Block block : blocks) {
            features.add(Clause.parse(block.toClause()));
        }

        table = new int[countVertices(dataset)][blocks.size()];

        int vertIndex = 0;
        dataset.reset();
        while (dataset.hasNextExample()) {
            Example ex = dataset.nextExample();

            Map<String, Integer> vertices = indexVertices(ex);
            Collection<String> sortedNames = vertices.keySet().stream().sorted().toList();

            for (String vertexName : sortedNames) {
                int vertexId = vertices.get(vertexName);

                int featIndex = 0;
                for (Block block : blocks) {

                    if (TuplesSettings.COUNT_GROUNDINGS) {
                        // TODO
                        table[vertIndex][featIndex] = -1;
                    } else {
                        if (block.literalDomain(ex).integerSet().contains(vertexId)) {
                            table[vertIndex][featIndex] = 1;
                        } else {
                            table[vertIndex][featIndex] = 0;
                        }
                    }

                    featIndex++;
                }

                vertIndex++;
            }
        }
    }

    @Override
    public String toString() {
        return "FeaturesTable{\n" +
                printFeatures() + '\n' +
                Arrays.deepToString(table) + "\n}";
    }

    private String printFeatures() {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (Clause feature : features) {
            sb.append(feature);
            sb.append("; ");
        }
        sb.append(']');
        return sb.toString();
    }

    private int countVertices(Dataset dataset) {
        int count = 0;

        dataset.reset();

        while (dataset.hasNextExample()) {
            Example ex = dataset.nextExample();

            for (int litId : ex.literalIDs()) {
                Literal lit = ex.integerToLiteral(litId);
                if (lit.predicate().equals(TuplesSettings.VERTEX_PREDICATE_NAME)) {
                    count++;
                }
            }
        }

        return count;
    }

    private Map<String, Integer> indexVertices(Example ex) {
        Map<String, Integer> retVal = new HashMap<>();

        for (int litId : ex.literalIDs()) {
            Literal lit = ex.integerToLiteral(litId);
            if (lit.predicate().equals(TuplesSettings.VERTEX_PREDICATE_NAME)) {
                String vertexName = lit.terms().iterator().next().name();
                retVal.put(vertexName, litId);
            }
        }

        return retVal;
    }

}
