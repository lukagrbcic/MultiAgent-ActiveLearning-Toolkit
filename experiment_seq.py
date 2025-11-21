import numpy as np
import llm_query as llmq
from analyze_performance import algorithm_analysis
import generate_algorithm as ga
import sys

sys.path.insert(0, 'ActiveLearningExperiment')
sys.path.insert(1, 'ActiveLearningExperiment/src')
sys.path.insert(2, 'ActiveLearningExperiment/src/samplers')
sys.path.insert(3, 'code')


class llm_discovery:
    def __init__(self, initial_query, iterations, orig_performance):
        self.initial_query = initial_query
        self.iterations = iterations
        self.orig_performance = orig_performance

    def llm_code(self, query):
        """Send query to LLM and return generated code."""
        llm = llmq.LLM(query)
        response = llm.get_response()
        return llm.get_code(response)

    def performance_summary(self, p_score, error):
        """Generate structured performance feedback for the LLM, ignoring speed."""
        if error:
            return {
                "status": "error",
                "message": f"Code execution failed: {error}",
                "suggestion": "Fix runtime errors before optimizing performance.",
                "metrics": None
            }

        change_perf = round((p_score - self.orig_performance) / self.orig_performance * 100, 2)

        if p_score > self.orig_performance:
            status = "success"
            suggestion = "Performance improved! Try to push it even further."
        elif p_score == self.orig_performance:
            status = "no_change"
            suggestion = "Performance stayed the same. Consider changing the algorithm."
        else:  # p_score < baseline
            status = "failure"
            suggestion = "Performance decreased. Identify causes and try a new approach."

        return {
            "status": status,
            "message": f"Performance change: {change_perf}%",
            "suggestion": suggestion,
            "metrics": {
                "baseline_score": self.orig_performance,
                "new_score": p_score
            }
        }

    def modify_query(self, previous_code, feedback):
        """Update query with previous code and structured feedback."""
        structured_feedback = (
            f"\nPrevious code:\n{previous_code}\n"
            f"Feedback:\nStatus: {feedback['status']}\n"
            f"Message: {feedback['message']}\n"
            f"Suggestion: {feedback['suggestion']}\n"
            f"Metrics: {feedback['metrics']}\n"
        )
        return self.initial_query + structured_feedback

    def generate_run(self, code_to_run):
        """Generate and run code, then return performance metrics."""
        ga.generate(code_to_run).write_code()
        return algorithm_analysis()

    def check_if_error(self, performance):
        """Return structured error metrics if execution failed."""
        if performance[0] == -1:
            return -1, performance[1] if len(performance) > 1 else None
        return performance[0], None  # performance[1] = runtime but we ignore it

    def loop(self):
        """Main improvement loop focusing only on performance."""
        code = self.llm_code(self.initial_query)
        print('Running initial code...')
        performance = self.generate_run(code)
        score, error = self.check_if_error(performance)

        results = {
            'code': [code],
            'perf_score': [score]
        }

        feedback = self.performance_summary(score, error)

        for i in range(self.iterations):
            print(f'\nIteration {i+1}')
            query = self.modify_query(code, feedback)
            code = self.llm_code(query)
            performance = self.generate_run(code)
            score, error = self.check_if_error(performance)

            results['code'].append(code)
            results['perf_score'].append(score)

            feedback = self.performance_summary(score, error)
            print(f"Status: {feedback['status']} | {feedback['message']}")

        np.save('code.npy', results)
        return results
