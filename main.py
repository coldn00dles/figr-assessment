'''
Brief Explanation of the Design : The agent as a whole relies on a memory object, which is in the form of a dictionary, and a base LLM which is
invoked when the user submits a query. Conversational memory is a part of the invoked request, and the LLM is instructed accordingly in its system prompt
to process it and the question by the user properly to formulate an answer. The agent generates code snippets as well as test cases, and also
responds to the user's counter questions if needed.

Inside the Script : It utilises the Llama 3.1 model as the base LLM, hosted locally through Ollama. The Agent has a method for invoking
function calls to the underlying LLM, which has been made using a custom Modelfile containing parameters and its system prompt. It also has a retry mechanism
which exhausts upon 5 continous retries. 
On-disk memory for the agent was implemented through a simple load-and-store mechanism using JSON. The model is provided the last 7 conservations along with the user's question, and is also updated
with each new conversation. Reasoning for choosing the Llama 3.1 model include its 128k token context window (for incorporating memory properly) and it being a SOTA open source model.

Things that werent implemented : Better logic for selecting conversations from memory for invokation (involving embeddings for getting most relevant results) and more individual LLM function calling methods that would ensure smoother experience. 

Assumptions :
1) Given that Ollama has been configured properly, local environment is sufficient to run
2) No external frameworks are used to build the agent.'''


import requests
import json
import regex as re 
import uuid
import time



class Memory:
    def __init__(self):
        with open("memory.json", "r") as file:   #loading the on-disk memory file
            memory = json.load(file)
        self.memory = memory

    def add_entry(self, user_question: str, assistant_response: str) -> None:
        entry = {"id": str(uuid.uuid4()), "entry": [
                {
                    "role": "user", "content": user_question   #stores a specific conversation to the memory in the given format
                },
                {
                    "role": "assistant", "content": assistant_response
                }
            ]}
        self.memory["history"].append(entry)
    
    def store_to_disk(self) -> None:    #storing the memory to disk 
        with open("memory.json", "w") as file:
            json.dump(self.memory, file)

    def prompt_format(self, question: str) -> str:  #formats the final question prompt to be given to the LLM
        formatted_prompt = ""
        if self.memory["history"]:  
            last_five_entries = self.memory["history"][-7:] if len(self.memory["history"]) >= 7 else self.memory["history"] #last seven conversational history items in their append order
            formatted_prompt = "History : \n"
            for entry in last_five_entries:
                for message in entry["entry"]:
                    formatted_prompt += f"{message['role']}: {message['content']}\n" 
        else:  
            formatted_prompt += f"user question: {question}\nassistant response: "    #formatting the history and the question asked by user
        return formatted_prompt

mem = Memory()

class CodeAgent:

    def __init__(self):
        self.model_name = "codeagent-llama"  #name of ollama model made using custom modelfile
        self.retry_count = 0
        self.max_retries = 5    #retry mechanism allows max 5 retries at a time

    def run(self, question: str) -> dict:           #serves as main call method for the agent
        while self.retry_count < self.max_retries:
            try:
                res = self.base_call(question)
                if res["response"] == "Error":
                    raise(Exception)
            except Exception as e:
                self.retry_count += 1
                print(f"Retrying...")
                time.sleep(0.2)
            else:
                self.retry_count = 0  # retry counter back to 0 upon no error
                return res["response"]
        print("Max retries hit!")
            

    def base_call(self, question: str) -> dict:          #invoke request to LLM
        url = "http://localhost:8080/api/generate"       #ollama hosted endpoint
        formatted_prompt = mem.prompt_format(question)
        payload = {
        "model": self.model_name,
        "prompt": formatted_prompt,         
        "stream" : True                         #stream = True allows for streamable chunk responses
        }
        headers = {
            "Content-Type": "application/json"
        }
        full_response = ""
        response = requests.post(url, json=payload, headers=headers, stream=True)
    
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)   #streamable printing
                    chunk = json_response.get("response", "")
                    full_response += chunk
                    print(chunk, end="", flush=True)  
            
            print()
            mem.add_entry(question, full_response)  #add the conversation to memory
            mem.store_to_disk()   #save memory to disk
            return {"response": full_response}
        else:
            return {"response": "Error"}


agent = CodeAgent()

def execute_test_case():
    ques_one = "Generate code in Python for printing Fibonacci series upto 6 numbers"
    ques_two = "Yes, I would like to modify the code to be in recursive."
    ques_three = "Modify the code to print upto 8 numbers."
    response_one = agent.run(question = ques_one)
    response_two = agent.run(question = ques_two)
    response_three = agent.run(question = ques_three)

if __name__ == "__main__":
    execute_test_case()
