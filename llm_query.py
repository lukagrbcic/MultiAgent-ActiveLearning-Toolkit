import numpy as np
import openai
import os
import re as re
import generate_algorithm as ga



client = openai.OpenAI(
    api_key = os.environ.get('CBORG_API_KEY'),
    base_url = "https://api.cborg.lbl.gov"
)

class LLM:

    def __init__(self, query, model, temperature=1.0):

        self.query = query
        self.model = model
        self.temperature = temperature
        print ('using model:', self.model)
        
    def get_response(self):
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages = [
                    {
                        "role": "user",
                        "content": self.query
                    }
                ],
                temperature=self.temperature
            )
                
            model_response = response.choices[-1].message.content
            
            return model_response
        except:
            print(f"Error calling model {self.model}")
    
    def get_code(self, response):
        
        sections = response.split("```")

        extracted_code = sections[1].replace('python', '')
        
        return extracted_code
        

                


