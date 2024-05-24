/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils;

import java.io.*;
import java.util.*;

/**
 * A class for easier creation of command-line utilities. It contains several useful functions.
 * 
 * @author admin
 */
public class CommandLine {

    /**
     * Parses parameters set on the command line in the following form:
     * -arg1 value of argument one -arg2 value of argument2 -arg3 "there is - here"
     * @param args the arguments as obtained in main(String args[])
     * @return Map containing arguments and their values, usage e.g. map.get("-arg1");
     */
    public static Map<String,String> parseParams(String[] args){
        Map<String,String> map = new HashMap<String,String>();
        StringBuilder sb = new StringBuilder();
        String paramName = null;
        for (String s : args){
            if (s.length() > 0 && s.charAt(0) == '-'){
                if (paramName != null){
                    map.put(paramName, sb.toString().trim());
                    sb = new StringBuilder();
                }
                paramName = s;
            } else {
                sb.append(" "+s);
            }
        }
        if (paramName != null){
            map.put(paramName, sb.toString().trim());
        } else {
            map.put("#", sb.toString().trim());
        }
        return map;
    }

    /**
     * Reads user input from command line.
     * @return read string
     */
    public static String read(){
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line = null;
        try {
            if ((line = br.readLine()) != null){
                return line;
            } else {
                return null;
            }
        } catch (IOException  ioe){
            ioe.printStackTrace();
            return null;
        }
    }
}
