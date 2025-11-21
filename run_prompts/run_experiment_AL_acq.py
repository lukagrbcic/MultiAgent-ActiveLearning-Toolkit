
from experiment import llm_discovery
import os

if os.environ.get("DEV_MODE") == "1":
    sys.dont_write_bytecode = True

initial_query = """This a code I use in my sampling loop with Random Forests. It is a greedy batched active learning sampling approach.

            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            from _PSO import PSO

            class modelSampler:

                def __init__(self, X, y, sample_size, lb, ub):

                    self.X = X
                    self.y = y
                    self.sample_size = sample_size
                    self.lb = lb
                    self.ub = ub

                    self.model = RandomForestRegressor().fit(self.X, self.y)

                def gen_samples(self):

                    X = []
                    f = []
                    for i in range(self.sample_size):

                        def get_values(X_population):

                            X_population = np.atleast_2d(X_population)
                            all_preds = np.array([
                                tree.predict(X_population) for tree in self.model.estimators_
                            ])
                            stds_across_trees = np.std(all_preds, axis=0)  # shape: (n_particles, output_feature_length)
                            particle_fitness = -np.sum(stds_across_trees, axis=1)  # shape: (n_particles,)

                            return particle_fitness

                        opt = PSO(function=get_values, lb=self.lb, ub=self.ub, swarm_size=10, max_evals=100, device='cpu')
                        min_x, _, min_f = opt.search()
                        min_x = np.ravel(min_x)

                        X.append(min_x)
                        f.append(min_f[0])

                    X = np.array(X)
                    f = np.array(f)

                    X = X[np.argsort(f)]

                    return X, self.model

            # =============================================================================
            # Active Learning Sampler Implementation — LLM Prompt
            # =============================================================================

            WORKFLOW CONTEXT:
            -----------------
            - The `modelSampler` class is part of an iterative ACTIVE LEARNING LOOP:
                1. Call `gen_samples()` to select a batch of new samples within bounds `lb` and `ub`.
                2. Add these samples into `self.X` (features) and `self.y` (labels).
                3. Retrain the model with the updated datasets.
            - This process repeats many times — the acquisition/model combination must improve accuracy cumulatively over several iterations, not just in a single step.
            - The problem is HIGH-DIMENSIONAL: e.g., expanding 3D inputs to 822D outputs in **multioutput regression**.

            CURRENT BEHAVIOR:
            -----------------
            - `gen_samples()` currently:
                - Returns exactly `sample_size` rows, matching the feature count defined by `len(lb)` and `len(ub)`.
            - Sampling is guided by a custom **Particle Swarm Optimization (PSO)** to locate points with highest estimated uncertainty.
            - The current regression model is a Random Forest, trained and evaluated externally for accuracy.

            CONSTRAINTS (Non-Negotiable):
            -----------------------------
            1. Keep the class name `modelSampler` and all existing method names.
            2. Preserve exact input/output behavior of `gen_samples()`:
                - Must return an array of size `(sample_size, n_features)` with correct types/shapes.
            3. Preserve the meaning and role of `lb` and `ub` as feature bounds.
            4. Implement **one** acquisition/model combination at runtime (no optional placeholders).
            5. Do not use pure random sampling or Latin Hypercube Sampling.
            6. The PSO optimizer must return a batch of fitness values.
            7. Avoid “fake” performance gains from simply increasing PSO iterations.
            8. Code must be fully compatible with the existing active learning workflow.
            9. Any new model must support `.fit` and `.predict` and provide robust uncertainty estimates for multioutput regression.

            GOALS:
            ------
            - Implement a single, **innovative batched active learning acquisition strategy**.
            - The model should be quick to train / evaluate.
            - The solution must minimize labeled data requirements while maximizing performance in the active learning loop.
            - Consider **changing the regression model** if stagnation occurs — the current Random Forest may not be optimal for high-dimensional output.
            - Adapt or hybridize acquisition functions:
                - Uncertainty sampling
                - Diversity sampling
                - Information-theoretic selection
                - Classic sampling algorithms.
            - The combination of acquisition + model must work efficiently for 822D multioutput regression.
            - The algorithm must yield consistent iterative performance improvements over multiple active learning loop iterations.
            - Code must remain clear, well-commented, and computationally efficient.

            SEED IDEAS (You may choose one or combine several):
            ---------------------------------------------------

            - **Hybrid uncertainty + diversity**: Select high-uncertainty samples with spatial diversity, using Determinantal Point Processes (DPP) or clustering penalties.
            - **Model ensembles**: Aggregate predictions from multiple regressors to estimate per-output variance, guide PSO toward impactful candidates.
            - **Multioutput-aware uncertainty**: Reduce or combine per-output uncertainties using PCA or weighted norms.
            - **Adaptive strategy**: Begin with diversity-heavy exploration, transition to pure exploitation (uncertainty sampling) as the model stabilizes.
            - **Information gain maximization**: Predict expected model change or mutual information for candidate points during PSO optimization.
            - **Hybrid sampling**: Sobol and similar samplers combined with active learning.

            TASK:
            -----
            Using the workflow context, constraints, goals, and seed ideas above:
            1. Produce **one complete Python class** `modelSampler` that:
                - Implements your chosen innovative batched acquisition strategy.
            2. The acquisition/model combination must integrate seamlessly with the existing PSO or any other optimizer for candidate point selection.
            3. Include inline comments explaining *why* each change is expected to improve accuracy or efficiency.
            4. Output **exactly one complete code block** — no placeholders, no multiple blocks.
            5. Ensure compatibility with the existing active learning loop (I/O, method names, bounds handling).


            OUTPUT FORMAT:
            --------------
            - One Python code block containing the full `modelSampler` class.
            - Inline comments explaining the reasoning behind each significant change.
            # =============================================================================

            """

run_llm = llm_discovery(initial_query, 300, 0.79886)
result = run_llm.loop()


#claude haiku legacy RUNNING
#grok mini legacy
#gpt oss legacy
#gpt oss deepthought legacy

#claude haiku best
#grok haiku best
#gpt oss best
#gpt oss deepthought best




