package ida.tuples;

import ida.ilp.basic.Clause;
import ida.ilp.basic.Literal;
import ida.ilp.treeLiker.*;
import ida.ilp.treeLiker.aggregables.GroundingCountingAggregablesBuilder;
import ida.ilp.treeLiker.aggregables.VoidAggregablesBuilder;
import ida.utils.collections.IntegerSet;

import java.util.*;

/**
 * Wrapper class for all TUPLES-related calls to TreeLiker.
 */
public class TreeLikerGateway {

    /**
     * Construct treeliker features for the dataset using TuplesSettings.TEMPLATE_DEPTH parameter.
     *
     * @param dataset symbolic examples
     * @return constructed features along with a table of their evaluations
     */
    public static FeaturesTable buildTreeLikerTable(List<Clause> dataset) {
        return buildTreeLikerTable(dataset, TuplesSettings.TEMPLATE_DEPTH);
    }

    /**
     * Construct treeliker features for the dataset.
     *
     * @param dataset      symbolic examples
     * @param featureDepth max depth of the constructed features
     * @return constructed features along with a table of their evaluations
     */
    public static FeaturesTable buildTreeLikerTable(List<Clause> dataset, int featureDepth) {
        TreeLikerGateway gateway = new TreeLikerGateway();
        return gateway.buildTable(dataset, featureDepth);
    }

    /**
     * Construct treeliker features for the dataset using the provided template.
     *
     * @param dataset  symbolic examples
     * @param template language bias
     * @return constructed features along with a table of their evaluations
     */
    public static FeaturesTable buildTreeLikerTable(List<Clause> dataset, String template) {
        TreeLikerGateway gateway = new TreeLikerGateway();
        return gateway.buildTable(dataset, template);
    }







    /**
     * Construct treeliker features for the dataset using TuplesSettings.TEMPLATE_DEPTH parameter.
     *
     * @param dataset symbolic examples
     * @return constructed features along with a table of their evaluations
     */
    public FeaturesTable buildTable(List<Clause> dataset) {
        return buildTable(dataset, TuplesSettings.TEMPLATE_DEPTH);
    }

    /**
     * Construct treeliker features for the dataset.
     *
     * @param dataset      symbolic examples
     * @param featureDepth max depth of the constructed features
     * @return constructed features along with a table of their evaluations
     */
    public FeaturesTable buildTable(List<Clause> dataset, int featureDepth) {
        PreprocessedInput preprocessed = preprocess(dataset, featureDepth);
        return constructAndEvaluateFeatures(preprocessed);
    }

    /**
     * Construct treeliker features for the dataset using the provided template.
     *
     * @param dataset  symbolic examples
     * @param template language bias
     * @return constructed features along with a table of their evaluations
     */
    public FeaturesTable buildTable(List<Clause> dataset, String template) {
        PreprocessedInput preprocessed = new PreprocessedInput(template, dataset);
        return constructAndEvaluateFeatures(preprocessed);
    }






    public FeaturesTable buildTableForExplanations(List<Clause> dataset) {
        return buildTableForExplanations(dataset, TuplesSettings.TEMPLATE_DEPTH);
    }

    public FeaturesTable buildTableForExplanations(List<Clause> dataset, int featureDepth) {
        PreprocessedInput preprocessed = preprocess(dataset, featureDepth);
        return constructAndEvaluateFeaturesForExplanations(preprocessed);
    }

    public FeaturesTable buildTableForExplanations(List<Clause> dataset, String template) {
        PreprocessedInput preprocessed = new PreprocessedInput(template, dataset);
        return constructAndEvaluateFeaturesForExplanations(preprocessed);
    }


    private PreprocessedInput preprocess(List<Clause> dataset, int depth) {
        GraphTemplateBuilder templateBuilder = new GraphTemplateBuilder(dataset);
        String stringTemplate = templateBuilder.inferTemplate(depth);
        return new PreprocessedInput(stringTemplate, dataset);
    }

    private FeaturesTable constructAndEvaluateFeatures(PreprocessedInput preprocessed) {
        Set<Block> features = constructFeatures(preprocessed);
        return evaluateFeatures(features, preprocessed.getGlobalConstants(), preprocessed.getDataset());
    }

    private FeaturesTable constructAndEvaluateFeaturesForExplanations(PreprocessedInput preprocessed) {
        Set<Block> features = constructFeatures(preprocessed);
        return FeaturesTable.forGraphExplanations(features, preprocessed.getDataset());
    }

    private Set<Block> constructFeatures(PreprocessedInput preprocessed) {
        Set<Block> features = new HashSet<>();

        for (Set<PredicateDefinition> def : preprocessed.getTemplates()) {
            HiFi hifi = new HiFi(preprocessed.getDataset());
//            hifi.setMaxSize(Integer.MAX_VALUE);
            if (TuplesSettings.COUNT_GROUNDINGS) {
                hifi.setAggregablesBuilder(GroundingCountingAggregablesBuilder.construct());
                hifi.setPostProcessingAggregablesBuilder(GroundingCountingAggregablesBuilder.construct());
            }

            features.addAll(hifi.constructFeatures(def));
        }

        return features;
    }

    private FeaturesTable evaluateFeatures(Set<Block> features, List<PredicateDefinition> globalConstants, Dataset dataset) {
        Table<Integer, String> table = new Table<>();

        List<Block> nonConstantAttributes = new ArrayList<Block>();
        List<Block> globalConstantAttributes = new ArrayList<Block>();
        for (Block attribute : features) {
            if (attribute.definition().isGlobalConstant()) {
                globalConstantAttributes.add(attribute);
            } else {
                nonConstantAttributes.add(attribute);
            }
        }

        Dataset copyOfDataset = dataset.shallowCopy();
        copyOfDataset.reset();

        while (copyOfDataset.hasNextExample()) {
            Example example = copyOfDataset.nextExample();
            table.addClassification(copyOfDataset.currentIndex(), copyOfDataset.classificationOfCurrentExample());
            addGlobalConstants(example, copyOfDataset.currentIndex(), table, globalConstants);
        }

        HiFi hifi = new HiFi(dataset);
        if (TuplesSettings.COUNT_GROUNDINGS) {
            hifi.setAggregablesBuilder(VoidAggregablesBuilder.construct());
            hifi.setPostProcessingAggregablesBuilder(GroundingCountingAggregablesBuilder.construct());
        }

        table.addAll(hifi.constructTable(nonConstantAttributes));
        return FeaturesTable.fromTreeLikerTable(table);
    }

    private void addGlobalConstants(Example example, int exampleIndex, Table<Integer, String> t, List<PredicateDefinition> globalConstants) {
        for (PredicateDefinition def : globalConstants) {
            IntegerSet domain = example.getLiteralDomain(def.predicate());
            //otherwise it can't be considered a global-constant-feature
            if (domain.size() == 1) {
                int literalId = domain.values()[0];
                Literal literal = example.integerToLiteral(literalId);
                for (int i = 0; i < def.modes().length; i++) {
                    if (def.modes()[i] == PredicateDefinition.GLOBAL_CONSTANT) {
                        if (literal.arity() == 1) {
                            t.add(exampleIndex, def.stringPredicate(), literal.get(i).toString());
                            //System.out.println("adding global constant: "+exampleIndex+" -- "+def.stringPredicate()+" "+literal.get(i).toString());
                        } else {
                            System.out.println("Warning: " + def + " cannot be used as global constant because its arity is not equal to one!!!!");
                        }
                    }
                }
            } else {
                if (domain.size() > 1) {
                    System.out.println("Warning: " + def + " cannot be used as global constant because there are more than one literals of the kind " + def + " in the example being processed or there is none!!!!");
                } else {
                    System.out.println("Warning: " + def + " cannot be used as global constant because there are no literals of the kind " + def + " in the example being processed or there is none!!!!");
                }
            }
        }
    }

}
