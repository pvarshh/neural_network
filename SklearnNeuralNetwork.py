from sklearn.neural_network import MLPClassifier

class SklearnNeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # activation='logistic' corresponds to Sigmoid
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_size,), 
            activation='logistic', 
            solver='sgd', 
            learning_rate_init=0.1,
            max_iter=1, # We just want to initialize weights
            random_state=42
        )
        
    def train_dummy(self, X, y):
        """Initialize weights by fitting on dummy data"""
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict_proba(X)
