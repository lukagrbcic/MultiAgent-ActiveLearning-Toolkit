import numpy as np

# result = np.load('code.npy', allow_pickle=True).item()
result = np.load('code.npy', allow_pickle=True).item()


# Get the 'code' list from the dict
code_list = result['code']

# Select the element at index 1
code_str = code_list[1]

# Save it to a .py file
with open('model_sampler.py', 'w', encoding='utf-8') as f:
    f.write(code_str)