# Use different models to infer the same problem alternatively
import json
import numpy as np
from vllm import LLM, SamplingParams

def get_completion(model, user_prompt):
    from openai import OpenAI
    client = OpenAI()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]
    )
    print(completion.choices[0].message)


data_path = "test_filtered.jsonl"
with open(data_path, "r") as f:
    data = [json.loads(line) for line in f]

# Load the model
checkpoint_path = "models/meta-llama/Llama-3.2-1B-Instruct"
model_1 = LLM(model=str(checkpoint_path), tokenizer = checkpoint_path, swap_space=8)

sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=1024, 
        n = 1
    )

for i in range(5):
    user_prompt = data[i]["problem"]
    # use the first model to infer
    output = model_1.generate(user_prompt, sampling_params)
    print(output)
    import time
    time.sleep(1010)