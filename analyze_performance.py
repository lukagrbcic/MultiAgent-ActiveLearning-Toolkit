import numpy as np
import matplotlib.pyplot as plt   
import sys
import json  # Import the json library
sys.path.insert(0, 'ActiveLearningExperiment')
sys.path.insert(1, 'ActiveLearningExperiment/src')
sys.path.insert(2, 'ActiveLearningExperiment/src/samplers')
sys.path.insert(3, 'code')


from run_AL import get_performance

#def algorithm_analysis():

#    score = get_performance()

#    return score

#score = algorithm_analysis()
#print(score)





def algorithm_analysis():
    # Your robust get_performance() will return either (score, True) or (-1, "error message")
    score, result_data = get_performance()

    # Create a dictionary to hold the result
    output_data = {
        "status": "success" if score != -1 else "error",
        "score": score,
        "details": result_data  # This will be True on success, or the error string on failure
    }
    return output_data

#if __name__ == "__main__":
result = algorithm_analysis()

print(json.dumps(result))

        
        
        


