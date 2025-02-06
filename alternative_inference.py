# Use different models to infer the same problem alternatively
import json
import numpy as np
from vllm import LLM, SamplingParams
from math_utils.grader import grade_answer

LLM1_TEMPLATE = \
"""
You are a helpful assistant. Solve the problem step by step. Use the exact step format:
- Each step: `## Step X:` (where X is 1, 2, 3, ...)
At the end, provide a concise numeric result in the format: `Final Answer: <your_answer>`, which should be a single number.

Problem:
{problem}

Solution:
## Step 1:
"""

LLM2_TEMPLATE = \
"""
You are a helpful assistant. The user has partially solved the problem, but there may be mistakes. 
Please review the provided solution steps carefully. Continue the solution step by step in the same format:
`## Step X:` 
At the end, write a concise numeric result in the format: `Final Answer: <your_answer>`, which should be a single number.

Problem:
{problem}

Solution:
{solution}
"""
def get_completion(model, user_prompt):
    from openai import OpenAI
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return completion.choices[0].message.content

def parse_answer(answer: str):
    # lower case
    answer = answer.lower()
    if "final answer:" in answer:
        return answer.split("final answer:")[1].strip(". \n")
    else:
        return answer.split(":")[-1].strip(". \n")
    return answer


data_path = "test_filtered.jsonl"
with open(data_path, "r") as f:
    data = [json.loads(line) for line in f]

# Load the model
checkpoint_path = "models/meta-llama/Llama-3.2-3B-Instruct"
model_1 = LLM(model=str(checkpoint_path), tokenizer = checkpoint_path, swap_space=8)

sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=512, 
        n = 1
    )

correct = 0
acc_1 = 0
acc_2 = 0
cutoff = 30
for i in range(30, cutoff + 30):
    user_prompt = data[i]["problem"]
    # use the first model to infer
    output = model_1.generate(LLM1_TEMPLATE.format(problem=user_prompt), sampling_params)
    #print("\n" + output[0].outputs[0].text + "\n")
    total_step = 1
    while f"## Step {total_step + 1}:" in output[0].outputs[0].text:
        total_step += 1
    #print("total_step: ", total_step)
    #print("parse_answer: ", parse_answer(output[0].outputs[0].text))
    #print("grade_answer: ", grade_answer(data[i]["solution"], parse_answer(output[0].outputs[0].text)))
    #print("grade_answer: ", grade_answer("15", parse_answer(output[0].outputs[0].text)))
    print("correct answer: ", data[i]["answer"], "inferred answer: ", parse_answer(output[0].outputs[0].text))
    if grade_answer(data[i]["answer"], parse_answer(output[0].outputs[0].text)):
        correct += 1
        continue

    # use the second model:
    #   1. without the prefix
    #   2. with the prefix (the first two steps)
    output_1 = get_completion("gpt-4o", LLM2_TEMPLATE.format(problem=user_prompt, solution=""))
    output_2 = get_completion("gpt-4o", LLM2_TEMPLATE.format(problem=user_prompt, solution=output[0].outputs[0].text.split(f"## Step 3:")[0]))

    acc_1 += grade_answer(data[i]["answer"], parse_answer(output_1))
    acc_2 += grade_answer(data[i]["answer"], parse_answer(output_2))
    print("correct answer: ", data[i]["answer"], "inferred answer: ", parse_answer(output_1))
    print("correct answer: ", data[i]["answer"], "inferred answer: ", parse_answer(output_2))
print("correct percentage: ", correct / cutoff)
print("acc_1: ", acc_1 / (cutoff - correct))
print("acc_2: ", acc_2 / (cutoff - correct))