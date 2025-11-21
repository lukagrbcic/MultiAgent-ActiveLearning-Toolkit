
from experiment import llm_discovery
import os

if os.environ.get("DEV_MODE") == "1":
    sys.dont_write_bytecode = True

initial_query = """This a code I use in my batched active learning loop with Random Forests. It is the current best performing code.
                   The input data is 3D (X), while the output data is 822D (y).

            import numpy as np
            from indago import PSO
            import random
            from sklearn.ensemble import RandomForestRegressor

            class modelSampler:

                def __init__(self, X, y, sample_size, lb, ub, function='uncertainty'):

                    self.X = X
                    self.y = y
                    self.sample_size = sample_size
                    self.lb = lb
                    self.ub = ub
                    self.function = function

                    self.model = RandomForestRegressor().fit(self.X, self.y)

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

                    return X, self.model


            Current behavior:
            1. For each candidate point in the batch (`sample_size`), the optimizer searches within bounds `lb` and `ub` to maximize uncertainty.
            2. The uncertainty score is computed by calculating the per-output standard deviation across predictions from `self.model.estimators_`, summing the components, and returning the negative value (since the optimizer minimizes by default).
            3. The final returned matrix `X` has `sample_size` rows and the same number of columns as features (length of `lb` and `ub`).
            4. The model is also returned to be later assessed for accuracy.
            5. The model is initially trained with the sampled dataset self.X and self.y.

            Constraints:
            - Do not change the names of the class (`modelSampler`) or its methods, as they are tied to other code.
            - Do not alter the return format of `get_samples()`.
            - Keep overall functionality identical (same inputs/outputs).
            - Leave `lb` and `ub` interpretation unchanged.
            - Keep the optimizer as is if you're using optimization.

            Goal:
            - Use innovative batched active learning acquisition strategies.
            - Experiment with novel or hybrid acquisition functions for active learning.
            - The sampling algorithm should yield an accurate model by using the minimum number of samples.
            - Determine the best model for this high dimensional active learning problem (3D -> 822D!).
            - The model must work well with the active learning approach, i.e. have a robust uncertainty estimation.
            - The model must have .fit and .predict scikit-learn type methods.
            - Keep code clear and maintainable.

            Deliverables:
            Please output the full improved code while:
            1. Preserving functionality and naming.
            2. Adding in-line comments explaining each performance improvement made.
            3. Generate the full Python code, nothing in chunks.
            """

run_llm = llm_discovery(initial_query, 10, 0.638)
result = run_llm.loop()


#claude haiku legacy RUNNING
#grok mini legacy
#gpt oss legacy
#gpt oss deepthought legacy

#claude haiku best
#grok haiku best
#gpt oss best
#gpt oss deepthought best




