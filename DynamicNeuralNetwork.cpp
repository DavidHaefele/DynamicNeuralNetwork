#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

#define EPOCHS 1000
#define LEARNINGRATE 0.1

int correct = 0;

struct Data
{
    vector<vector<double>> samples;
    Data(int numSamples, int numFeatures)
    {
        for(int i = 0; i < numSamples; i++)
        {
            vector<double> features;
            for(int j = 0; j < numFeatures; j++)
                features.push_back(0.0);

            samples.push_back(features);
        }
    }
};

struct Layer
{
    vector<double> nodes, errors, deltas, weights, biases;
    vector<int> targets;

    Layer(int numNodes)
    {
        cout << "new Layer with " << numNodes << " nodes created" << endl;
        for(int i = 0; i < numNodes; i++)
        {
            nodes.push_back(0.0);
            errors.push_back(0.0);
            deltas.push_back(0.0);
            targets.push_back(0.0);
            biases.push_back((double) rand() / RAND_MAX * 2.0 - 1.0);
        }
    }

    void setWeights(int numWeights)
    {
        srand(time(NULL));
        for(int i = 0; i < numWeights; i++)
            weights.push_back((double) rand() / RAND_MAX * 2.0 - 1.0);
    }

};

vector<Layer> Layers;

double sigmoid(double x)
{
    return 1 / (1 + exp(-1 * x));
}

double sigmoid_prime(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

int getMaxIndexOutput()
{
    //computes index of output node with highest value
    int numLayers = Layers.size() - 1;
    int numOutputNodes = Layers[numLayers].nodes.size();
    int index = 0;

    for(int i = 1; i < numOutputNodes; i++)
        if(Layers[numLayers].nodes[i] > Layers[numLayers].nodes[index])
            index = i;

    return index;
}

int countLines(FILE *ptr)
{
    fseek(ptr, 0, SEEK_SET);
    unsigned int numberOfLines = 0;
    int ch;

    while(EOF != (ch = getc(ptr)))
        if('\n' == ch)
            ++numberOfLines;

    return numberOfLines;
}

int countFeatures(FILE *ptr)
{
    //count training values/features per line
    fseek(ptr, 0, SEEK_SET);
    unsigned int numberOfFeatures = 0;
    int ch;

    while(EOF != (ch = getc(ptr)) && ch != '\n')
        if(',' == ch)
            ++numberOfFeatures;

    return numberOfFeatures;
}

Data readData(const char* filename)
{
    FILE *infile = fopen(filename, "r");
    int numLines = countLines(infile);
    int numFeatures = countFeatures(infile)+1;
    fseek(infile, 0, SEEK_SET);
    Data data(numLines, numFeatures);
    char chr;
    string feature;
    int lineCount = 0;
    int featureCount = 0;

    while(fread(&chr, sizeof(chr), 1, infile)){
        if(chr != ',' && chr != '\n')
            feature += chr;

        else
        {
            const char* c = feature.c_str();
            data.samples[lineCount][featureCount++] = atof(c);
            feature = "";
        }

        if(featureCount == numFeatures)
        {
            lineCount++;
            featureCount = 0;
        }
    }
    fclose(infile);

    //adjusting labels to start from 0
    for(int i = 0; i < numLines; i++)
        data.samples[i][data.samples[i].size()-1] -= 1;

    return data;
}

void normalize(Data data)
{
    //turns every feature into a value between 0 and 1
    int numFeatures = data.samples[0].size();
    int numSamples = data.samples.size();
    double maxValue = data.samples[0][0];

    for(int i = 0; i < numSamples; i++)
        for(int j = 0; j < numFeatures - 1; j++)
            if(data.samples[i][j] > maxValue)
                maxValue = data.samples[i][j];

    for(int i = 0; i < numSamples; i++)
        for(int j = 0; j < numFeatures - 1; j++)
            data.samples[i][j] /= maxValue;
}

int feedForward(Data data, int currentSample)
{
    //returns index of maximal output value
    //initalize input layer
    int numInputNodes = Layers[0].nodes.size();
    int numFeatures = data.samples[0].size();
    for(int i = 0; i < numInputNodes; i++)
        for(int j = 0; j < numFeatures - 1; j++)
            Layers[0].nodes[j] = data.samples[currentSample][j];

    int numLeftNodes, numRightNodes;
    int numLayers = Layers.size();

    //feed forward from hidden layers to output layer
    for(vector<Layer>::size_type i = 1; i != numLayers; i++)
    {
        numLeftNodes = Layers[i-1].nodes.size();
        numRightNodes = Layers[i].nodes.size();
        for(int j = 0; j < numRightNodes; j++)
        {
            for(int k = 0; k < numLeftNodes; k++)
            {
                if(i > 1)
                    Layers[i].nodes[j] += sigmoid(Layers[i-1].nodes[k]) * Layers[i-1].weights[j * numLeftNodes + k];
                else
                    Layers[i].nodes[j] += Layers[i-1].nodes[k] * Layers[i-1].weights[j * numLeftNodes + k];
            }
            Layers[i].nodes[j] += Layers[i].biases[j];
        }
    }

    int prediction = getMaxIndexOutput();
    if(prediction == data.samples[currentSample][numFeatures - 1])
        correct++;

    return prediction;
}

void backPropagate()
{
    int numLayers = Layers.size() - 1;
    int numOutputNodes = Layers[numLayers].nodes.size();

    //calculate errors and error derivatives for nodes of output layer
    for(int i = 0; i < numOutputNodes; i++)
    {
        Layers[numLayers].errors[i] = Layers[numLayers].targets[i] - sigmoid(Layers[numLayers].nodes[i]);
        Layers[numLayers].deltas[i] = Layers[numLayers].errors[i] * sigmoid_prime(Layers[numLayers].nodes[i]);
    }

    int numLeftNodes, numRightNodes;
    //calulate errors and error derivatives for nodes of hidden layers
    for (unsigned i = numLayers; i > 1; i--)
    {
        numLeftNodes = Layers[i-1].nodes.size();
        numRightNodes = Layers[i].nodes.size();
        for(int j = 0; j < numLeftNodes; j++)
        {
            Layers[i-1].errors[j] = 0.0;
            for(int k = 0; k < numRightNodes; k++)
            {
                    Layers[i-1].errors[j] += Layers[i-1].weights[k * numRightNodes + j] * Layers[i].deltas[k];
            }
            Layers[i-1].deltas[j] = Layers[i-1].errors[j] * sigmoid_prime(Layers[i-1].nodes[j]);
        }
    }
}

void updateWeights()
{
    int nodesOfLeft = 0;
    int nodesOfRight = 0;
    int numLeftNodes, numRightNodes;

    for (unsigned i = Layers.size() - 1; i > 0; i--)
    {
        numLeftNodes = Layers[i-1].nodes.size();
        numRightNodes = Layers[i].nodes.size();
        for(int j = 0; j < numLeftNodes * numRightNodes; j++)
        {
            nodesOfLeft = j % numLeftNodes;

            if(j % numLeftNodes == 0 && j != 0)
                nodesOfRight++;

            //calculation of weight change in left layer depending on error derivatives of right layer and values of left layer
            if(i > 1)
                Layers[i-1].weights[j] += LEARNINGRATE * Layers[i].deltas[nodesOfRight] * sigmoid(Layers[i-1].nodes[nodesOfLeft]);
            else
                Layers[i-1].weights[j] += LEARNINGRATE * Layers[i].deltas[nodesOfRight] * Layers[i-1].nodes[nodesOfLeft];
        }

        for(int j = 0; j < numRightNodes; j++)
            Layers[i].biases[j] += LEARNINGRATE * Layers[i].deltas[j];
    }
}

void train(Data data)
{
    int numLayers = Layers.size() - 1;
    int numOutputNodes = Layers[numLayers].nodes.size();
    int numFeatures = data.samples[0].size();
    int numSamples = data.samples.size();
    int label = numFeatures - 1;

    for(int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double cost = 0.0;
        for(int sample = 0; sample < numSamples; sample++)
        {
            //reset node values
            for(int i = 0; i < numLayers + 1; i++)
                fill(Layers[i].nodes.begin(), Layers[i].nodes.end(), 0.0);

            //assign targets to corresponding output nodes
            for (int i = 0; i < numOutputNodes; i++)
                Layers[numLayers].targets[i] = ((int)data.samples[sample][label] == i) ? 1 : 0;

            feedForward(data, sample);

            //compute cost/error in current epoch
            for(int i = 0; i < numOutputNodes; i++)
                cost += pow(Layers[numLayers].targets[i] - sigmoid(Layers[numLayers].nodes[i]), 2);

            //reset node deltas
            for(int i = 0; i < numLayers + 1; i++)
                fill(Layers[i].deltas.begin(), Layers[i].deltas.end(), 0.0);

            backPropagate();
            updateWeights();
        }
        cout << "EPOCH\t" << epoch << "\t| cost = " << cost << "  \t| correct = " << correct << endl;
        correct = 0;
    }
}

void initLayers(int numNodes[], int numLayers)
{
    for(int i = 0; i < numLayers; i++)
        Layers.push_back(Layer(numNodes[i]));

    for(vector<Layer>::size_type i = 0; i != numLayers - 1; i++)
        Layers[i].setWeights(Layers[i].nodes.size() * Layers[i+1].nodes.size());
}

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        cout << "usage: ./DynamicNeuralNetwork [train_data.csv] [number of neurons in layer 1] [number of neurons in layer 2] ..." << endl;
        return -1;
    }

    else
    {
        int numNodes[argc - 2];
        for(int i = 2; i < argc; i++)
            numNodes[i-2] = atoi(argv[i]);

        Data data = readData(argv[1]);
        normalize(data);
        initLayers(numNodes, argc - 2);
        train(data);
    }

    return 0;
}
