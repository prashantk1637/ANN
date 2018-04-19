/*
L = Learning Rate
M = Momentum
N = Training Time or Epochs
H = Hidden Layers
*/
package MLP;

/**
 *
 * @author Prashant
 */
import weka.core.Instances;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
public class MLP implements Serializable
{ 
    public static void main(String args[]) throws Exception{
	       //ArffLoader loader=new ArffLoader();
              
               CSVLoader loader=new CSVLoader();
               loader.setSource(new File("C:/Users/Prashant/Documents/MLPData.csv"));
		Instances trainingdata =loader.getDataSet();
               trainingdata.setClassIndex(trainingdata.numAttributes()-1);
               
               // Weka MLP  Call 
               MultilayerPerceptron mlp=new MultilayerPerceptron();
              
               mlp.setLearningRate(0.1);
               mlp.setTrainingTime(2000); // No of epoch
               mlp.setHiddenLayers("10"); //No of nodes in hidden layer
               //mlp.setGUI(true);
               
               mlp.buildClassifier(trainingdata);
               //Print Training data evaluation details like error
               Evaluation eval = new Evaluation(trainingdata);
               eval.evaluateModel(mlp, trainingdata);
                System.out.println("Error="+eval.errorRate()); //Printing Training Mean root squared Error
               System.out.println(eval.toSummaryString()); //Summary of Training 
               
               //Storing mlp intance variable 
              try{
               FileOutputStream fos=new FileOutputStream(new File("C:\\Users\\Prashant\\Documents\\mlpObject.txt"));
               ObjectOutputStream oos=new ObjectOutputStream(fos);
               oos.writeObject(mlp);
               oos.flush(); 
                }catch(Exception e){}
              
              
              
             
               
               System.out.println("===========================Testing========================");
              
               //ArffLoader loader1=new ArffLoader();
               CSVLoader loader1=new CSVLoader();
               loader1.setSource(new File("C:/Users/Prashant/Documents/MLPTestData.csv"));
               Instances testdata =loader1.getDataSet();
              testdata.setClassIndex(testdata.numAttributes()-1); 
               //Reading mlp object
               
               try{
                     FileInputStream fis=new FileInputStream(new File("C:\\Users\\Prashant\\Documents\\mlpObject.txt"));
                    ObjectInputStream ois=new ObjectInputStream(fis);
                         MultilayerPerceptron storedMLP=(MultilayerPerceptron)ois.readObject();
                           System.out.println(storedMLP);
                           
                         
               
                        for(int i=0;i<testdata.numInstances();i++)
                        {
                            double actualClass=testdata.instance(i).classValue();
                            double predictedClass=storedMLP.classifyInstance(testdata.instance(i));
                            if(predictedClass>0.5)
                                predictedClass=1;
                            else predictedClass=0;
                            System.out.println("Actual Class="+actualClass+" ,"+"Predicted Class="+predictedClass);
                            
                        }
              
               }catch(Exception e){}
              
              
              
             
             
              
               
		
	}
 

}
