import java.util.*;
class GeneraalizedSingleLayerANN
{
      	static double E=0;
	static double learning_rate=0.1,theta=0.1; //fixed for this case
            static int itr=0,max_itr=1000;
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

	static void perceptron(double x[][], double w[][], int y[][],double e[], int n, int d)
	{
                double u[][]=new double[n][2];
                double v[][]=new double[n][2];
                
                while(itr<max_itr) // max 1000 epoch
		{	E=0;
                       
                    for(int i=0;i<n;i++)
                    {    e[i]=0;
                        for(int j=0;j<2;j++)
                        {   u[i][j]=0;
                            for(int k=0;k<d+1;k++)
                            {
                               u[i][j]=u[i][j]+x[i][k]*w[k][j];
                            }
                            v[i][j]=sigmoid(u[i][j]);
                            e[i]=e[i]+0.5*(y[i][j]-v[i][j])*(y[i][j]-v[i][j]);
                            
                         }
                           E=E+e[i];
                           // update w[i][j]
                                for(int p=0;p<d+1;p++)
                                {   for(int q=0;q<2;q++)
                                        w[p][q]=w[p][q]+learning_rate*(y[i][q]-v[i][q])*diff_sigmoid(u[i][q])*x[i][p];
                                       
                                }
                           
                    }
                    if(E/n<theta)
				break;
                itr++; 
                }
                       
		



	}
	public static void main(String[] args)
	{
            Scanner sc=new Scanner(System.in);
            System.out.println("Enter number of data points");
            int n=sc.nextInt();
            System.out.println("Enter number of features");
           int  d=sc.nextInt();
            //class
            int y[][]=new int[n][2];
            //data
          double x[][]= new double[n][d+1];
          double w[][]= new double[d+1][2];
          double e[]=new double[n];
          
          //generate random data;
          for(int i=0;i<n;i=i+2)
          {     x[i][0]=1; // fixed  it will be used with bias weight
                y[i][0]=1;
                y[i][1]=0;
                 for(int j=1;j<d+1;j++)
                        x[i][j]=RandomNumber(0,1); 
             
                
                 x[i+1][0]=1; // fixed  it will be used with bias weight
                 y[i+1][0]=0;
                 y[i+1][1]=1;
                 for(int j=1;j<d+1;j++)
                        x[i+1][j]=RandomNumber(1,2); 
                  
             
          }
           
          for(int i=0;i<d+1;i++)
          {
              w[i][0]=RandomNumber(0,2);
              w[i][1]=RandomNumber(0,2);
          }
           
          perceptron(x,w,y,e,n,d);
          //System.out.println("E(avg)="+E/n);
          double accuracy=(1-E/n)*100;
            System.out.println("Accuracy="+accuracy+"%");
          
          ///Test a point
          double sum1=0,sum2=0;
          double p[]=new double[d+1];// 1 point having d features for testing p[0]=1; fixed
          p[0]=1;
          for(int i=1;i<d+1;i++)
               p[i]=RandomNumber(1, 2);
          for(int i=0;i<d+1;i++)
              sum1=sum1+p[i]*w[i][0];
          for(int i=0;i<d+1;i++)
              sum2=sum2+p[i]*w[i][1];
          
          if(sum1<sum2)
                System.out.println("class 1--->good classification");
          else System.out.println("class 0--->bad classification");
          
             
          
                    
                   
        }
         
}
