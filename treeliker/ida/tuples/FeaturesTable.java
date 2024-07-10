package ida.tuples;

import ida.ilp.basic.Clause;
import ida.ilp.treeLiker.Table;

import java.util.*;

public class FeaturesTable {

    public final List<Clause> features;
    public final int[][] table;

    public FeaturesTable(Table<Integer, String> propTable) {
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

    @Override
    public String toString() {
        return "FeaturesTable{\n" +
                printFeatures() + '\n' +
                Arrays.deepToString(table) + "}\n";
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

}
