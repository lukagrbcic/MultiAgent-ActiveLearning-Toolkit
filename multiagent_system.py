import numpy as np
from experiment import llm_discovery


# run_llm = llm_discovery(initial_query, 100, 0.630, model) # 10, 10, 100 setup
# result = run_llm.loop()


class multi_agent_system:

    def __init__(self, initial_prompt, n_agents, epochs, baseline_score, 
                    elite_archive_size=4,
                    agent_iterations=20,
                    exploit_patience=3,
                    explore_interval=10,
                    agent_model='lbl/cborg-coder:latest',
                    agent_idea_model='lbl/cborg-coder:latest',
                    evaluator='local'):
        
        self.initial_prompt = initial_prompt
        self.n_agents = n_agents
        self.epochs = epochs
        self.baseline_score = baseline_score
        self.elite_archive_size = elite_archive_size
        self.agent_iterations = agent_iterations
        self.exploit_patience = exploit_patience
        self.explore_interval = explore_interval
        self.agent_model = agent_model
        self.agent_idea_model = agent_idea_model
        self.evaluator = evaluator
        
    
    
    
        
    
    
    




