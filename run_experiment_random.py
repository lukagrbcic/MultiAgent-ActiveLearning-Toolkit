
from experiment import llm_discovery
import os
import sys

if os.environ.get("DEV_MODE") == "1":
    sys.dont_write_bytecode = True

initial_query = """This a code I use in my sampling loop with Random Forests. It is a random sampling approach at the moment.

            import numpy as np
            import random
            from sklearn.ensemble import RandomForestRegressor

            np.random.seed(random.randint(0, 10223))


            class modelSampler:

                def __init__(self,  X, y, sample_size, lb, ub):

                    self.X = X
                    self.y = y
                    self.sample_size = sample_size
                    self.lb = lb
                    self.ub = ub

                    self.model = RandomForestRegressor().fit(self.X, self.y)

                def gen_samples(self):

                    n = self.sample_size
                    d = len(self.lb)
                    sample_set = np.array([np.random.uniform(self.lb, self.ub, d) for i in range(n)])

                    return sample_set, self.model


            # =============================================================================
            # Active Learning Sampler Implementation
            # =============================================================================
            #
            # WORKFLOW CONTEXT:
            # -----------------
            # This `modelSampler` class is part of a larger ACTIVE LEARNING LOOP.
            # The loop operates iteratively as follows:
            #   1. Call `gen_samples()` to select a new set of samples within bounds `lb` and `ub`.
            #   2. Add these samples to the datasets `self.X` (features) and `self.y` (labels).
            #   3. Retrain the model using the updated `self.X` and `self.y`.
            #
            # This process repeats many times — meaning the sampling algorithm must aim to:
            #   - Improve the model not just in the current iteration, but OVER MULTIPLE FUTURE ITERATIONS.
            #   - Select batches that maximize cumulative accuracy across the loop.
            #
            # The acquisition strategy must be effective in HIGH-DIMENSIONAL problems
            # (e.g., expanding 3D input to 822D features).
            #
            # -----------------------------------------------------------------------------
            #
            # CURRENT BEHAVIOR:
            # -----------------
            # 1. `gen_samples()` returns a matrix `sample_set` with:
            #       - Exactly `sample_size` rows
            #       - Number of columns == length of `lb` and `ub`
            # 2. `self.X` and `self.y` are datasets gathered so far in the active learning loop.
            # 3. A model is initially trained on `self.X` and `self.y`.
            # 4. The model is returned later for accuracy assessment.
            #
            # -----------------------------------------------------------------------------
            #
            # CONSTRAINTS:
            # ------------
            # - Keep the class name `modelSampler` and all method names unchanged (external code depends on them).
            # - Preserve the exact return format and interface of `gen_samples()`.
            # - `sample_size` is fixed and should not be changed. The returned `sample_set` must have `sample_size` number of rows.
            # - Maintain the meaning of `lb` and `ub` as feature bounds.
            # - Do NOT implement multiple strategies — only ONE algorithm should exist and be active at runtime.
            # - Input/output behavior must remain exactly the same (shape, types).
            # - Preserve full compatibility with the existing active learning workflow.
            # - Do not use standalone random or any kind of sampling.
            #
            # -----------------------------------------------------------------------------
            #
            # GOALS:
            # ------
            # - Implement a single, innovative BATCHED ACTIVE LEARNING acquisition strategy.
            #     ("Batched" = selecting multiple samples at once per iteration.)
            # - Can adapt known acquisition functions (e.g., uncertainty sampling, diversity sampling)
            #   or combine them creatively in a hybrid approach.
            # - Minimize the number of samples required to reach high accuracy.
            # - Model must have `.fit` and `.predict` methods (scikit-learn style) AND provide robust uncertainty estimates.
            # - Code should be clear, maintainable, and computationally efficient for high-dimensional datasets.
            # - The algorithm must consistently contribute to iterative improvements in the active learning loop.
            #
            # -----------------------------------------------------------------------------
            #
            # DELIVERABLES:
            # -------------
            # 1. Full Python code for the improved `modelSampler` (output as one complete code block).
            # 2. Inline comments describing WHY each change improves performance or accuracy.
            # 3. The algorithm implemented in `gen_samples()` must be directly used — no placeholders or optional code paths.
            # 4. Code must be ready to run in the described active learning loop without interface changes.
            # 5. Always respond ONLY with Python code inside a fenced code block like this:
            #        ```python
            #        # code here
            #        ```
            #
            # =============================================================================

            """


#model='google/gemini-flash'
#model='anthropic/claude-haiku'
#model='openai/o4-mini'
#model='openai/o3-mini'
#model='openai/gpt-5-mini'
model='lbl/cborg-coder:latest'
#model='lbl/cborg-deepthought:latest'
#model = 'xai/grok-mini'


run_llm = llm_discovery(initial_query, 100, 0.630, model) # 10, 10, 100 setup
result = run_llm.loop()





