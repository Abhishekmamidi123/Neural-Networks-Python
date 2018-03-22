# ===  CODE IS WORKING  === #
 
# input_layer_size = 5 units
# hidden_layer_size = 4 units
# num_labels = 1 unit(ouput layer)
# m 

# Input, Hidden, Output, Iterations, Learning_rate.

# =================== numpy functions =======================
# k=np.array([1,2,3],dtype='float32')
# np.exp(l)
# np.arange(0,10,2)
# np.dot(d,e)  or  d.dot(e)  --> matrix multiplication of d and e.
# +,-,/,*                    --> Element-wise operation.
# a.shape
# np.zeros(shape=(x,y))
# np.random.randn(1,3)
# np.c_[ A, np.ones(N) ]     --> insert a column
# np.r_[ A, [A[1]] ]	     --> insert a row
# np.transpose(x)
# np.delete(a,0,1)           --> delete first column
# np.delete(a,1,0)           --> delete second column

import numpy as np

def sigmoid(z) :
	return 1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z) :
	return sigmoid(z) * (1.0 - sigmoid(z))

class NeuralNetwork(object):	
	# Function to initialize
	def __init__(self, x, hidden, ly, iterations, learningRate):
		#            x : Number of units of hidden layer
		#       hidden : Number of units of hidden layer
		#            y : Number of units of output layer
		#   iterations : Number of iterations
		# learningRate : (for updating weights)
		self.x = x
		self.hidden = hidden
		self.ly = ly
		self.iterations = iterations
		self.learningRate = learningRate
		# Initialize weights randomly to avoid symmetry.
		self.m = 0
		
		# self.theta1  : Weights between input layer and hidden layer.
		# self.theta2  : Weights between hidden layer and Output layer.
		self.theta1 = np.random.randn(self.hidden, 1+self.x)
		self.theta2 = np.random.randn(self.ly, 1+self.hidden)
		self.h = np.zeros(shape=(y.shape[0],1))
		self.y = np.zeros(shape=(y.shape[0],1))
	
	# Train neural network
	def trainData(self, X, y):
		self.m = X.shape[0]
		for i in range(self.iterations):
			
			# Forward propagation
			# activation function of input layer which is equal to x(input) - a1
			a1 = np.c_[np.ones(shape=(self.m,1)), X]
			z2 = np.dot(a1, np.transpose(self.theta1))
			# activation function of hidden layer which is equal to x(input) - a2
			a2 = np.c_[np.ones(shape=(z2.shape[0],1)), sigmoid(z2)]
			z3 = np.dot(a2, np.transpose(self.theta2))
			# activation function of output layer which is equal to x(input) - a3
			a3 = sigmoid(z3)
			h = a3
			
			# Backward propagation
			# Find delta of predicted values 'h' and real output 'y'
			delta3 = h-y
			# delta of hidden layer - delta2
			delta2 = np.dot(delta3,self.theta2)*sigmoidGradient(np.c_[np.ones(shape=(z2.shape[0],1)), z2])
			delta2 = np.delete(delta2,0,1)
			# We don't calculate delta1(as it is input layer)
			
			Delta1 = np.dot(np.transpose(delta2),a1)
			Delta2 = np.dot(np.transpose(delta3),a2)
			
			# Gradient of Theta
			Theta1_gradient = Delta1/self.m
			Theta2_gradient = Delta2/self.m
			
			# Gradient Descent update(we have gradient)
			self.thata1 = self.theta1-(self.learningRate)*(Theta1_gradient)
			self.theta2 = self.theta2-(self.learningRate)*(Theta2_gradient)
			
			# findError
			error=0
			if y.shape[1]==1 :    # one unit in output layer
				error = (sum(h-y))/self.m
			else :		      # multiple units in output layer
				error = sum(sum(h-y))/self.m
				
		self.h = h      # Predicted output
		self.y = y      # Output
		print self.h
		print self.y
		
	# Predict output
	def predictOutput(self, predict_x):
		# Perform forward propagation and predict output.
		predict_x = np.transpose(predict_x)
		a1 = np.r_[[np.array([1],'float32')], predict_x]
		z2 = np.dot(self.theta1,a1)
		a2 = np.r_[[np.array([1],'float32')], sigmoid(z2)]
		z3 = np.dot(self.theta2,a2)
		a3 = sigmoid(z3)
		predict_output = np.transpose(a3)
		return predict_output
		
# Load Data
# Just for example-data
X = np.array([[1,2,3,1,2],[2,3,1,2,3],[2,1,3,2,1],[2,1,3,1,2],[3,2,1,2,2],[3,1,2,3,2],[1,3,2,1,4]],dtype='float32')
y = np.array([[1,0],[0,1],[1,0],[1,0],[1,0],[0,1],[1,0]],dtype='float32')

# Normalise Data
X_norm = (X - X.min(0)) / X.ptp(0)

# trainData
# (input units,hidden units, output units, no. of iterations, learningRate)
Neural = NeuralNetwork(5,4,2,100000,0.01)
Neural.trainData(X,y)

# predictOutput
predict_x = np.array([[2,3,1,2,3]])
prediction = Neural.predictOutput(predict_x)

# printOutput
print prediction
