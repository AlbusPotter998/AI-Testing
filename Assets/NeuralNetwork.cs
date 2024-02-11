using JetBrains.Annotations;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuralNetwork : MonoBehaviour
{
    
    // Start is called before the first frame update
    public struct Node
    {
        public double[] weights;
        public double[] biases;
        public double findActivation(double[] inputs, double[] weights, double[] biases)
        {
            double activation = 0;
            for(int i = 0; i < weights.Length; i++)
            {
                activation += (inputs[i] * weights[i]) + biases[i];
            }
            return 1 / (1 + Math.Exp(-1 * activation));
        }
    }
    int layers = 5;
    public Node[][] network;
    public double[] finaloutputs;
    public int[] xpos;
    public int[] ypos;
    public bool[] poisonous;
    public double AverageError;
    public GameObject Point;
    public double output;
    public int trainingpoints = 200;
    double findOutput(double input)
    {
        return (input * input * input * input * input * input * input * 0.0000000000000013793) + (input * input * input * input * input * input * -0.0000000000022957) + (input * input * input * input * input * 0.0000000012496) + (input * input * input * input * -0.00000022673 + 539);
    }
    void createTrainingPoints(int pointnum)
    {
        var random = new System.Random();
        for (int i = 0; i < pointnum; i++)
        {
            int x = random.Next(700);
            int y = random.Next(700);
            GameObject clone = Instantiate(Point, new Vector3(x, y, 0), Quaternion.identity);
            if (findOutput(x) > y)
            {
                clone.GetComponent<Renderer>().material.SetColor("_Color", Color.blue);
                poisonous[i] = false;
            }
            else
            {
                clone.GetComponent<Renderer>().material.SetColor("_Color", Color.red);
                poisonous[i] = true;
            }

            xpos[i] = x;
            ypos[i] = y;

        }
    }
    public void GraphNetwork()
    {
        
        for (int x = 0; x < 800; x++)
        {
            int closetomidy = -1;
            double closediff = 2;
            for (int y = 1; y < 1000; y++)
            {
                double[] coordinates = new double[2];
                coordinates[0] = x;
                coordinates[1] = y;
                RunNetwork(coordinates);
                double probin = finaloutputs[0];
                double probp = finaloutputs[1];
                double diff = Math.Abs(Math.Sqrt(((0.5 - probin) * (0.5 - probin)) + ((0.5 - probp) * (0.5 - probp))));
                if (diff < closediff)
                {
                    closetomidy = y;
                    closediff = diff;
                }
                else
                {
                    Debug.Log(diff);
                }

            }
            GameObject clone = Instantiate(Point, new Vector3(x, closetomidy, 0), Quaternion.identity);
            clone.GetComponent<Renderer>().material.SetColor("_Color", Color.green);


        }
        
    }
    double ErrorCalculation() // This is specific to the configuration of this network. Change if using in a different project!
    {
        AverageError = 0;
        for(int i = 0; i < trainingpoints; i++)
        {
            double[] errorinputs = new double[] { xpos[i], ypos[i] };
            RunNetwork(errorinputs);
            if (poisonous[i] == true)
            {
                AverageError += Math.Abs(Math.Sqrt((finaloutputs[0] * finaloutputs[0]) + ((1 - finaloutputs[1]) * (1 - finaloutputs[1]))));
            }
            else
            {
                AverageError += Math.Abs(Math.Sqrt((finaloutputs[1] * finaloutputs[1]) + ((1 - finaloutputs[0]) * (1 - finaloutputs[0]))));
            }


        }
        return AverageError / trainingpoints;
    }

    void Start()
    {
        xpos = new int[trainingpoints];
        ypos = new int[trainingpoints];
        poisonous = new bool[trainingpoints];
        createTrainingPoints(trainingpoints);
        network = new Node[layers][];
        network[0] = new Node[2];
        network[1] = new Node[5];
        network[2] = new Node[5];
        network[3] = new Node[5];
        network[4] = new Node[2];
        finaloutputs = new double[network[network.Length - 1].Length];
        for(int i = 0; i < layers; i++)
        {
            for(int j = 0; j < network[i].Length; j++)
            {
                network[i][j] = new Node();
                if(i == 0)
                {
                    network[i][j].weights = new double[1];
                    network[i][j].biases = new double[1];
                }
                else
                {
                    network[i][j].weights = new double[network[i - 1].Length];
                    network[i][j].biases = new double[network[i - 1].Length];
                }
            }
        }
        
        Randomize();
        TrainNetwork(5, 1);
        /*
        for (int i = 0; i < 100; i++)
        {
            var random = new System.Random();
            double[] input = new double[] { random.Next(600), random.Next(1000) };
            RunNetwork(input);
            
        }
        */
        Debug.Log(ErrorCalculation());
        GraphNetwork();
    }
    void TrainNetwork(int steps, double trainingrate)
    {
        for(int i = 0; i < steps; i++)
        {
            for(int j = 0; j < layers; j++)
            {
                for(int k = 0; k < network[j].Length; k++)
                {
                    for (int l = 0; l < network[j][k].weights.Length; l++)
                    {
                        double oldweight = network[j][k].weights[l];
                        double oldbias = network[j][k].biases[l];
                        double minerror = ErrorCalculation();
                        double minweight = -200;
                        double minbias = -200;
                        network[j][k].weights[l] = -100;
                        network[j][k].biases[l] = -100;
                        for (double m = -100; m <= 100; m += trainingrate)
                        {
                            network[j][k].weights[l] = m;
                            double error = ErrorCalculation();
                            if(error < minerror)
                            {
                                minerror = error;
                                minweight = m;
                            }
                            
                        }
                        if(minweight == -200)
                        {
                            network[j][k].weights[l] = oldweight;
                        }
                        else
                        {
                            network[j][k].weights[l] = minweight;
                        }
                        
                        
                        minerror = ErrorCalculation();
                        for(double m = -100; m <= 100; m += trainingrate)
                        {
                            network[j][k].biases[l] = m;
                            double error = ErrorCalculation();
                            if(error < minerror)
                            {
                                minerror = error;
                                minbias = m;
                            }
                            
                        }
                        if(minbias == -200)
                        {
                            network[j][k].biases[l] = oldbias;
                        }
                        else
                        {
                            network[j][k].biases[l] = minbias;
                        }
                        
                        
                    }
                    
                }
            }
        }
    }
    void Randomize()
    {
        var random = new System.Random();
        for(int i = 0; i < layers; i++)
        {
            for(int j = 0; j < network[i].Length; j++)
            {
                for(int k = 0; k < network[i][j].weights.Length; k++)
                {
                    network[i][j].weights[k] = -100;
                    network[i][j].biases[k] = -100;
                }
                
            }
        }
    }
    public void RunNetwork(double[] goesin)
    {
        double[][] outputs = new double[layers][];
        for(int i = 0; i < layers; i++)
        {
            outputs[i] = new double[network[i].Length];
        }
        for(int i = 0; i < layers; i++)
        {
            double[] inputs;
            for (int j = 0; j < network[i].Length; j++)
            {
                
                if(i == 0)
                {
                    inputs = new double[1];
                    inputs[0] = goesin[j];
                }
                else
                {
                    inputs = new double[outputs[i - 1].Length];
                    outputs[i - 1].CopyTo(inputs, 0);
                }
                outputs[i][j] = network[i][j].findActivation(inputs, network[i][j].weights, network[i][j].biases);
            }
            if(i == layers - 1)
            {
                outputs[i].CopyTo(finaloutputs, 0);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
