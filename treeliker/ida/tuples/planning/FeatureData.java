package ida.tuples.planning;

import ida.ilp.basic.Clause;
import ida.ilp.treeLiker.Table;

import java.util.*;

public class FeatureData {

    public final List<Clause> features;
    public final int[][] table;

    public FeatureData(Table<Integer, String> propTable) {
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
        return "FeatureData {\n" +
                "\tfeatures:" + features.toString() + '\n' +
                "\ttable=" + tableToString() + '\n' +
                '}';
    }

    private String tableToString() {
        StringBuilder sb = new StringBuilder();
        for (int[] row : table) {
            sb.append(Arrays.toString(row));
            sb.append("; ");
        }
        return sb.toString();
    }
}
