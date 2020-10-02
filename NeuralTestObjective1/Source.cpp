#include <iostream>
#include <vector>

using namespace std;

class Layer;
class Network;

class Neuron {
private:
	float in;
	float out;
	float delta;
	vector <float> weights;

public: 
	Neuron() {
		this->in = 0;
		this->out = 0;
		this->delta = 0;
	}
	~Neuron() {}
	friend Layer;
	friend Network;
};

class Layer {
private:
	vector <Neuron> layer;

public:
	Layer(int n = 2) {
		this->layer.resize(n);
	}

	Layer(const Layer& x) {
		this->layer = x.layer;
	}

	Layer& operator=(const Layer& x) {
		if (&x == this)
			return *this;
		this->layer = x.layer;
	}

	friend Network;
};

class Network {
private:
	typedef vector <Neuron> layer;
	vector <layer> layers;

	typedef vector <float> bias_layer;
	vector <bias_layer> bias;

	bool biases;

	float learningRate;

	float error;
	float error_prev;

	static float sigmoid(float x) {
		return 1 / (1 + exp(-x));
	}

	float(*actFunc)(float x);

	static float stdGrad(float element, const float& gradient, const float& rate) {
		element += gradient * rate;
		return element;
	}

	float(*descentFunc)(float element, const float& gradient, const float& rate);

public:
	Network(Layer& In, vector <Layer> Hid, Layer& Out) {
		biases = true;

		actFunc = sigmoid;
		descentFunc = stdGrad;

		learningRate = 2;
		error = 0;
		error_prev = error;

		layers.resize(Hid.size() + 2);

		for (size_t i = 0; i != layers.size() - 2; i++) {
			layers[i + 1] = Hid[i].layer;
		}
		layers[0] = In.layer;
		layers[layers.size() - 1] = Out.layer;


		for (size_t i = 0; i != layers.size() - 1; i++) {
			for (size_t j = 0; j != layers[i].size(); j++) 
				layers[i][j].weights.resize(layers[i + 1].size(), 0.5);
		}

		if (biases) {
			bias.resize(layers.size() - 1);
			for (size_t i = 0; i != layers.size() - 1; i++)
				bias[i].resize(layers[i + 1].size(), 1);
		}

	}

	float process(float* arg, int argSize, int resultSize) {
		for (size_t i = 0; i != layers[0].size(); i++)
			layers[0][i].out = arg[i];

		for (size_t i = 1; i != layers.size(); i++)
			for (size_t j = 0; j != layers[i].size(); j++) {
				layers[i][j].in = 0;
				for (size_t k = 0; k != layers[i - 1].size(); k++)
					layers[i][j].in += layers[i - 1][k].out * layers[i - 1][k].weights[j];
				if (biases)
					layers[i][j].in += bias[i - 1][j];
				layers[i][j].out = actFunc(layers[i][j].in);
			}

		return layers[layers.size() - 1][0].out;
	}

	int correct(float* ideal, int resultSize) {
		float deltaSum;

		error_prev = error;
		error = 0;
		for (size_t i = 0; i != layers[layers.size() - 1].size(); i++)
			error += pow((ideal[i] - layers[layers.size() - 1][i].out), 2);
		error = sqrt(error);

		for (size_t i = 0; i != layers[layers.size() - 1].size(); i++)
			layers[layers.size() - 1][i].delta = (1 - layers[layers.size() - 1][i].out) * layers[layers.size() - 1][i].out
			* (ideal[i] - layers[layers.size() - 1][i].out);

		for (size_t i = layers.size() - 2; i != 0; i--)
			for (size_t j = 0; j != layers[i].size(); j++) {
				deltaSum = 0;
				for (size_t k = 0; k != layers[i + 1].size(); k++)
					deltaSum += layers[i + 1][k].delta * layers[i][j].weights[k];
				layers[i][j].delta = (1 - layers[i][j].out) * layers[i][j].out * deltaSum;
			}

		for (size_t i = 0; i != layers.size() - 1; i++)
			for (size_t j = 0; j != layers[i].size(); j++)
				for (size_t k = 0; k != layers[i + 1].size(); k++)
					layers[i][j].weights[k] = descentFunc(layers[i][j].weights[k], layers[i][j].out * layers[i + 1][k].delta, learningRate);

		if (biases) {
			for (size_t i = 0; i != layers.size() - 1; i++)
				for (size_t j = 0; j != layers[i + 1].size(); j++)
					bias[i][j] = descentFunc(bias[i][j], layers[i + 1][j].delta, learningRate);
		}
		return 0;
	}
 
	int learn(float** inputData, float** outputIdeal, int inputSize, int outputSize, int setSize, int generations) {
		for (size_t i = 0; i != generations; i++)
			for (size_t j = 0; j != setSize; j++) {
				process(inputData[j], inputSize, outputSize);
				correct(outputIdeal[j], outputSize);
			}
		return 0;
	}

	float getErrorDecrease() {
		return abs(this->error - this->error_prev);
	}

};


int main() {
	Layer In(2);
	//Layer Hid[1];
	vector <Layer> Hid;
	Layer Out(1);

	Hid.resize(1);
	Hid[0] = Layer(2);

	Network Hi(In, Hid, Out);
	float arg[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	float ideal[4][1] = { {0},{0},{1},{1} };

	float** arg1;
	arg1 = new float* [4];
	for (int i = 0; i < 4; i++)
		arg1[i] = arg[i];

	float** ideal1;
	ideal1 = new float* [4];
	for (int i = 0; i < 4; i++)
		ideal1[i] = ideal[i];

	for (int i = 0; i < 4; i++)
		cout << Hi.process(arg1[i], 2, 1) << endl;
	cout << endl;
	for (int i = 0; i < 1000; i++) {
		cout << Hi.getErrorDecrease() << endl;
		Hi.learn(arg1, ideal1, 2, 1, 4, 10);
	}
	cout << endl;
	for (int i = 0; i < 4; i++)
		cout << Hi.process(arg1[i], 2, 1) << endl;
	return 0;
}