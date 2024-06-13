      
import sys

import openai
openai.api_base = ""
openai.api_key = ''
import time
import json
import tqdm
from multiprocessing import Pool
openai.api_base = ""
openai.api_key = ''
gpt_model = 'gpt-3.5-turbo'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

system_prompt = '''
You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:
------
##INSTRUCTIONS:
- Focus on the meaningful match between the predicted answer and the correct answer.
- Consider synonyms or paraphrases as valid matches.
- Evaluate the correctness of the prediction compared to the answer.
'''

def judge(ele):
    template = '''Please evaluate the following video-based question-answer pair:
Question: {}
Correct Answer: {}
Predicted Answer: {}
If the predicted answer expresses the same meaning as the correct answer, please output 1; otherwise, output 0.
DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide 0 or 1.
'''
    gpt_judge = []

    prompt = template.format(ele['prompt'].replace("Answer with the option's letter from the given choices directly.", ""), ele['gt'], ele['pred'])
    max_retries = 20 
    retry_delay = 5  
    retries = 0
    output = None
    while output is None and retries < max_retries:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            output = openai.ChatCompletion.create(
                model=gpt_model,
                max_tokens=10,
                temperature=0,
                messages=messages)
            if output is not None:
                output = output['choices'][0]['message']['content']
            else:
                retries += 1
                print(f"Attempt {retries}: Failed to get response, retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)  
            print(f"An error occurred: {e}")
            retries += 1
            print(f"Attempt {retries}: Exception encountered, retrying after {retry_delay} seconds...")
            time.sleep(retry_delay) 
    if output is None:
        print("Failed to get a valid response from the API after maximum retries.")
        gpt_judge.append("No response")
    else:
        gpt_judge.append(output)
    print(output)
    ele['gpt_judge'] = gpt_judge
    return ele
import os
if __name__ == "__main__":
    output_file_path = args.output_file
    output_file = open(output_file_path, 'a')
    gpt_input = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    with Pool(150) as p:
        result = list(tqdm.tqdm(p.imap(judge, gpt_input), total=len(gpt_input)))
    for ele in result:
        output_file.write(json.dumps(ele)+"\n")

    