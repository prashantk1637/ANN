import java.util.*;
class SingleLayerANN
{
	//class
	static int y[][]=new int[100][2];
	//data
	static double x[][]= new double[100][3];

	static double w10,w11,w12,w20,w21,w22,E=0,e;
	static double learning_rate=0.1,theta=0.1; //fixed for this case
	static int itr=0,max_itr=1000;
	//rounding upto two decimal pla
	static double RandomNumber(int min,int max)
	{
		Random r=new Random();
		double num=min+r.nextDouble()*(max-min);
		return num;


	}
	static double diff_sigmoid(double val)
	{
		double diff_sig=(Math.exp(-val))/((1+Math.exp(-val))*(1+Math.exp(-val)));
		return diff_sig;
	}
	static double sigmoid(double val)
	{
		double sig=1.0/(1+Math.exp(-val));
		return sig;
	}

	static void perceptron()
	{

		while(itr<max_itr) // max 100 epoch
		{	E=0;
			for(int i=0;i<100;i++)
			{

				double u1=w10*x[i][0]+w11*x[i][1]+w12*x[i][2];                            /// for a data having 
				double u2=w20*x[i][0]+w21*x[i][1]+w22*x[1][2];                           /// two features 
				double v1=sigmoid(u1); 
				double v2=sigmoid(u2);
				e=((y[i][0]-v1)*(y[i][0]-v1)+(y[i][1]-v2)*(y[i][1]-v2));
				E=E+e;
				w10=w10+learning_rate*(y[i][0]-v1)*diff_sigmoid(u1)*x[i][0];
				w11=w11+learning_rate*(y[i][0]-v1)*diff_sigmoid(u1)*x[i][1];
				w12=w12+learning_rate*(y[i][0]-v1)*diff_sigmoid(u1)*x[i][2];
				w20=w20+learning_rate*(y[i][1]-v2)*diff_sigmoid(u2)*x[i][0];
				w21=w21+learning_rate*(y[i][1]-v2)*diff_sigmoid(u2)*x[i][1];
				w22=w22+learning_rate*(y[i][1]-v2)*diff_sigmoid(u2)*x[i][2];
			}
			if(E/100<theta)
				break;

		  itr++;

		}


	}
	public static void main(String[] args)
	{



		for(int i=0;i<100;i=i+2)
		{               // class 0 data
			x[i][0]=1; // fixed;
			x[i][1]=RandomNumber(0,1);
			x[i][2]=RandomNumber(0,1);
			y[i][0]=1; 
			y[i][1]=0;
                                      // class 1 data
			x[i+1][0]=1;
			x[i+1][1]=RandomNumber(1,2);
			x[i+1][2]=RandomNumber(1,2);
			y[i+1][0]=0;
			y[i+1][1]=1;
		}
		//Initial weights
		w10=RandomNumber(0,2);
		w11=RandomNumber(0,2);
		w12=RandomNumber(0,2);
		w20=RandomNumber(0,2);
		w21=RandomNumber(0,2);
		w22=RandomNumber(0,2);
		System.out.println("Class 0");
		for(int i=0;i<50;i++)
		{
			System.out.println("("+x[i][1]+","+x[i][2]+")");
		}
		System.out.println("Class 1");
		for(int i=50;i<100;i++)
		{
			System.out.println("("+x[i][1]+","+x[i][2]+")");
		}


			perceptron();
			System.out.println("E(avg)="+E/100);
                //testing        
                for(int i=0;i<5;i++)
		{
                    double x=RandomNumber(1, 2);
                    double y=RandomNumber(1, 2);
                    if(w10+w11*x+w12*y<w20+w21*w22*y)
                    {
                        
                        System.out.println("Class 1-->good classification");
                    }
                    else System.out.println("Class 0-->bad classification");
                }
	}
       




}
