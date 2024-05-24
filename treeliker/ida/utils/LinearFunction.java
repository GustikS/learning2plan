/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ida.utils;

/**
 * Class for representation of 
 * @author Ondra
 */
public class LinearFunction {
    
    //a0,a1,a2,...
    private double[] coeffs;
    
    /**
     * 
     * @param coeffs
     */
    public LinearFunction(double ...coeffs){
        this.coeffs = coeffs;
    }
    
    /**
     * 
     * @param values
     * @return
     */
    public double evaluate(double ...values){
        if (values.length != coeffs.length-1){
            throw new IllegalArgumentException();
        }
        double retVal = coeffs[0];
        for (int i = 0; i < values.length; i++){
            retVal += coeffs[i+1]*values[i];
        }
        return retVal;
    }
}
