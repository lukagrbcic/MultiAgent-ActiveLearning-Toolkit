import numpy as np
import random







str_ = """
import numpy as np
from indago import PSO
import random
from scipy.stats import norm

np.random.seed(random.randint(0, 10223))

class modelSampler:
    
    def __init__(self, model, sample_size, lb, ub, algorithm, function='expected_improvement'):
        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.function = function     

    def expected_improvement(self, x):
        preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)
        best_observed = np.min(mean_pred)
        z = (mean_pred - best_observed) / std_pred
        ei = (mean_pred - best_observed) * norm.cdf(z) + std_pred * norm.pdf(z)
        return -np.sum(ei)  # Negate because we want to maximize EI

    def get_samples(self):
        
        X = []
        f = []
        for i in range(self.sample_size):
            
            if self.function == 'expected_improvement':
                get_values = self.expected_improvement
            else:
                def get_values(x):               
                    preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
                    return -np.sum(np.std(preds, axis=0))
            
            optimizer = PSO()
            optimizer.evaluation_function = get_values 
            optimizer.lb = self.lb
            optimizer.ub = self.ub
            optimizer.max_evaluations = 100
            result = optimizer.optimize()
            min_x = result.X 
            min_f = result.f
            
            X.append(min_x)
            f.append(min_f)
                                
        X = np.array(X)
        f = np.array(f)
        
        X = X[np.argsort(f)]

        return X

            
"""


class generate:
    
    def __init__(self, new_code):
        
        self.new_code = new_code
        #self.code_orig = open('code/model_sampler.py', 'r').read()
    
    def write_code(self):
        
        new_code = open(f'ActiveLearningExperiment/src/samplers/model_sampler.py', 'w')
        for i in self.new_code:
            new_code.write(i)
        new_code.close()
        


# gen = generate(str_).write_code()



















