
from experiment import llm_discovery
import sys
sys.dont_write_bytecode = True

initial_query = """This a code I use in my batched active learning loop.

            import numpy as np
            from indago import PSO
            import random

            np.random.seed(random.randint(0, 10223))

            class modelSampler:

                def __init__(self, model, sample_size, lb, ub, algorithm, function='uncertainty'):

                    self.model = model
                    self.sample_size = sample_size
                    self.lb = lb
                    self.ub = ub
                    self.algorithm = algorithm
                    self.function = function

                def get_samples(self):

                    X = []
                    f = []
                    for i in range(self.sample_size):

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

            It has been tested and it works. The optimizer to determine the maximum uncertainty here is PSO (implemented in Indago module).

            The important thing is that the function returns a matrix that contains points that will be evaluated
            (self.sample_size rows and columns must match the lb and ub vectors).
            Do not change the names of the class/function as it is tied to other code. The values need to be returned from the get_samples function.

            Can you enhance the batched active learning algortihm to achieve a better model score and performance on a benchmark?
            Make sure that the batch yields the best improvement and is diverse enough. Feel free to also experiment with the acquisition.

            """

run_llm = llm_discovery(initial_query, 20, 2.96)
result = run_llm.loop()
