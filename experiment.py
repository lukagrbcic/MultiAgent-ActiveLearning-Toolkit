import numpy as np
import llm_query as llmq
import generate_algorithm as ga
import sys
import subprocess
import ast
import random
import json

class llm_discovery:
    def __init__(self, initial_query, iterations, orig_performance, model='lbl/cborg-coder:latest',
                 hard_reseed_interval=10, exploit_patience=3):

        self.initial_query = initial_query
        self.iterations = iterations
        self.orig_performance = orig_performance
        self.best_score = orig_performance
        self.best_code = None
        self.current_code = None
        self.model = model

        self.hard_reseed_interval = hard_reseed_interval
        self.exploit_patience = exploit_patience

    def llm_code(self, query):
        """Send query to LLM and return generated code."""
        llm = llmq.LLM(query, self.model)
        response = llm.get_response()
        return llm.get_code(response)

    def performance_summary(self, p_score, error):
        """Generate structured performance feedback for the LLM, now including best code/score."""
        if error:
            return {
                "status": "error",
                "message": f"Code execution failed: {error}",
                "suggestion": "Fix runtime errors before optimizing performance.",
                "metrics": None,
                "best_result": {
                    "best_score": self.best_score,
                    "best_code": self.best_code
                }
            }

        change_perf = round((p_score - self.orig_performance) / self.orig_performance * 100, 2)

        if p_score > self.best_score:
            self.best_score = p_score
            self.best_code = self.current_code

        if p_score > self.orig_performance:
            status = "success"
            suggestion = "Performance improved! Try to push it even further."
        elif p_score == self.orig_performance:
            status = "no_change"
            suggestion = "Performance stayed the same. Consider trying a different approach."
        else:
            status = "failure"
            suggestion = "Performance decreased. Try a fresh strategy."

        return {
            "status": status,
            "message": f"Performance change: {change_perf}%",
            "suggestion": suggestion,
            "metrics": {
                "baseline_score": self.orig_performance,
                "new_score": p_score
            },
            "best_result": {
                "best_score": self.best_score,
                "best_code": self.best_code
            }
        }

    def modify_query(self, previous_code, feedback, reason=None):
        """Update query with previous code and structured feedback."""

        if reason and "Improving top-performing" in reason:
            reason += (
                "\nIMPORTANT: Make only algorithmically meaningful modifications.\n"
                "- Alter the sample selection logic, acquisition function, or training process.\n"
                "- Avoid only changing constants, seeds, names, or comments.\n"
                "- Modifications must affect model behavior or performance."
            )

        structured_feedback = (
            (f"\nReason for choosing this code: {reason}\n" if reason else "")
            + f"\nPrevious code:\n{previous_code if previous_code else '[Starting fresh, no prior code]'}\n"
            + f"Feedback:\nStatus: {feedback['status']}\n"
            + f"Message: {feedback['message']}\n"
            + f"Suggestion: {feedback['suggestion']}\n"
            + f"Metrics: {feedback['metrics']}\n"
            + f"Best so far - Score: {feedback['best_result']['best_score']}\n"
            + f"Best Code:\n{feedback['best_result']['best_code']}\n"
        )
        return self.initial_query + structured_feedback

    def generate_run(self, code_to_run):
        """
        Generates and runs the candidate code, parsing the JSON output.
        This version uses a single, unified try/except block for cleaner error handling.
        """
        ga.generate(code_to_run).write_code()

        # Step 1: Execute the subprocess. This call itself is very unlikely to fail.
        # We capture the result without checking for errors yet.
        result = subprocess.run(
            [sys.executable, "analyze_performance.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Subprocess failed without a specific error message."
            print(f"--- SUBPROCESS ERROR ---")
            print(f"The script 'analyze_performance.py' crashed. Error: {error_message}")
            print("Full error output (stderr):\n" + result.stderr)
            print("------------------------")
            return (-1, f"Script execution failed: {error_message}")

        # Check 2: Did the script run successfully but produce no output?
        if not result.stdout.strip():
            print("--- NO OUTPUT ERROR ---")
            print("The script 'analyze_performance.py' ran successfully but produced no output.")
            print("This usually means a logical error prevented the final print statement from being reached.")
            print("-----------------------")
            return (-1, "Script produced no output.")

        # Check 3: If we have output, is it valid JSON?
        try:
            data = json.loads(result.stdout)
            score = data["score"]
            details = data.get("details", None)

            print("Performance Result:", (score, details))
            return (score, details)

        except json.JSONDecodeError:
            print("--- PARSING ERROR ---")
            print("The script ran but produced invalid JSON output.")
            print("Full script output (stdout):\n" + result.stdout)
            print("---------------------")
            return (-1, "Script produced invalid JSON output.")
        except KeyError as e:
            print("--- KEY ERROR ---")
            print(f"The script's JSON output is missing the required key: {e}")
            print("Full script output (stdout):\n" + result.stdout)
            print("-----------------")
            return (-1, f"JSON output missing required key: {e}")


    def check_if_error(self, performance):
        if performance[0] == -1:
            return -1, performance[1] if len(performance) > 1 else None
        return performance[0], None

    def loop(self):
        """Improvement loop with Elite Pool + Strategic Hints + Reseeds + True Exploration"""

        # === INITIAL RUN ===
        initial_code = self.llm_code(self.initial_query)
        self.current_code = initial_code
        print('Running initial code...')
        performance = self.generate_run(initial_code)
        score, error = self.check_if_error(performance)

        self.elites = [(score, initial_code)]
        self.best_score = score
        history = [(0, score, "initial", f"Performance: {score}")]

        results = {'code': [initial_code], 'perf_score': [score]}
        feedback = self.performance_summary(score, error)

        exploit_fail_count = 0
        mode = "exploit"

        # === MAIN LOOP ===
        for i in range(1, self.iterations + 1):
            print(f"\nIteration {i} [{mode.upper()} MODE]")

            # === Strategic hints / plateau detection ===
            strategic_hint = None
            if mode == "exploit" and exploit_fail_count >= self.exploit_patience:
                strategic_hint = (
                    "\nPERFORMANCE HAS PLATEAUED. "
                    "Try a significantly different approach to the acquisition strategy — "
                    "not just parameter tweaks. Consider novel heuristics, hybrid selection logic, "
                    "or unusual combinations of uncertainty/diversity."
                )

            # === Hard reseed on interval ===
            is_hard_reseed = (i % self.hard_reseed_interval == 0)
            if is_hard_reseed:
                print("💡 Hard reseed: starting fresh from initial prompt")
                base_code = None
                reason = "Hard reseed to escape local optima."
                mode = "explore"

            elif feedback['status'] == 'error':
                print("ERROR DETECTED — FIX MODE ACTIVATED.")
                base_code = self.current_code
                reason = "Fix runtime error before optimization."
                mode = "error_fix"

            elif mode == "exploit":
                base_code = max(self.elites, key=lambda x: x[0])[1]
                reason = "Improving top-performing code using incremental changes."
                if strategic_hint:
                    reason += strategic_hint

            elif mode == "explore":
                # True exploration: 50% fresh start, 50% random elite mutation
                if random.random() < 0.5:
                    base_code = None
                    reason = "Exploring from scratch with only constraints and goals."
                else:
                    base_code = random.choice(self.elites)[1]
                    reason = "Exploring by mutating a random elite."

            # === Build query ===
            prev_code_for_prompt = base_code if base_code else ""
            query = self.modify_query(prev_code_for_prompt, feedback, reason)

            history_text = "\n".join(
                [f"Iter {it}: score={sc:.5f} | status={st} | note={msg}"
                 for it, sc, st, msg in history[-5:]]
            )
            query += f"\nPerformance history (last {min(len(history), 5)}):\n{history_text}\n"

            # === Generate + Run new code ===
            code = self.llm_code(query)
            self.current_code = code
            performance = self.generate_run(code)
            score, error = self.check_if_error(performance)

            results['code'].append(code)
            results['perf_score'].append(score)

            # === New feedback ===
            feedback = self.performance_summary(score, error)
            print(f"Status: {feedback['status']} | {feedback['message']} | Best Score: {self.best_score}")

            # === History update ===
            history.append((i, score, feedback['status'], feedback['message']))

            # === Elite pool update ===
            if feedback['status'] != "error":
                self.elites.append((score, code))
                self.elites = sorted(self.elites, key=lambda x: x[0], reverse=True)[:5]

            # === Mode switching ===
            if mode == "error_fix" and feedback['status'] != "error":
                print("Error fixed. Returning to exploit mode.")
                mode = "exploit"
                exploit_fail_count = 0

            elif mode == "exploit":
                if score > self.best_score:
                    self.best_score = score
                    exploit_fail_count = 0
                    print("New best score found!")
                else:
                    exploit_fail_count += 1
                    if exploit_fail_count >= self.exploit_patience:
                        print("No improvement after exploit patience. Switching to explore...")
                        mode = "explore"
                        exploit_fail_count = 0

            elif mode == "explore" and score > self.best_score:
                print("Better score found during exploration. Switching back to exploit mode...")
                self.best_score = score
                mode = "exploit"
                exploit_fail_count = 0

        # === Save Best ===
        np.save('code.npy', results)
        with open("best_code.py", "w") as f:
            f.write(max(self.elites, key=lambda x: x[0])[1])

        return results
