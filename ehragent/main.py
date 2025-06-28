import os
import json
import random
import numpy as np
import argparse
import autogen
from toolset_high import *
from medagent import MedAgent,hf_inference_api_call
from config import openai_config, llm_config_list
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load these ONCE at program start (NOT in every loop!)
LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

print("Loading Llama-2 model... (this may take a minute)")
llama2_tokenizer = AutoTokenizer.from_pretrained(LLAMA2_MODEL_NAME)
llama2_model = AutoModelForCausalLM.from_pretrained(LLAMA2_MODEL_NAME)
# If you have a GPU, uncomment the next line
# llama2_model = llama2_model.to("cuda")

def llama2_local_inference(prompt, max_new_tokens=256):
    inputs = llama2_tokenizer(prompt, return_tensors="pt")
    # If using GPU: inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llama2_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=llama2_tokenizer.eos_token_id
        )
    result = llama2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def judge(pred, ans):
    old_flag = True
    if not ans in pred:
        old_flag = False
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "No" or ans == "no":
        ans = "0"
    if ans == "Yes" or ans == "yes":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if ", " in ans:
        ans = ans.split(', ')
    if ans[-2:] == ".0":
        ans = ans[:-2]
    if not type(ans) == list:
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if not ans[i] in pred:
            new_flag = False
            break
    return (old_flag or new_flag)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="<YOUR_LLM_NAME>")
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mimic_iii")
    parser.add_argument("--data_path", type=str, default="<YOUR_DATASET_PATH>")
    parser.add_argument("--logs_path", type=str, default="<YOUR_LOGS_PATH>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_id", type=str, default="521fd2885f51641a963f8d3e")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=4)
    args = parser.parse_args()
    set_seed(args.seed)

    config_list = [openai_config(args.llm)]
    config = config_list[0]
    llm_api_type = config.get("api_type", "").upper()

    if llm_api_type in ["OPENAI", "AZURE"]:
        llm_config = llm_config_list(args.seed, config_list)
        chatbot = autogen.agentchat.AssistantAgent(
            name="chatbot",
            system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code.",
            llm_config=llm_config,
        )
    else:
        chatbot = None
    
    #medagent setup
    user_proxy = MedAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding"},
        config_list=config_list,
    )

    # register the functions
    user_proxy.register_function(
        function_map={
            "python": run_code
        }
    )

    user_proxy.register_dataset(args.dataset)

    # read from json file
    with open(args.data_path, 'r') as f:
        contents = json.load(f)

    # random shuffle
    random.shuffle(contents)
    file_path = "{}/{}/".format(args.logs_path, args.num_shots) + "{id}.txt"

    start_time = time.time()
    if args.num_questions == -1:
        args.num_questions = len(contents)

    # ---- NEW: RAG-ready long_term_memory ----
    long_term_memory = []
    if args.dataset.lower().startswith("health"):  # e.g., "healthcaremagic"
        for item in contents:
            question = item.get("input", "")
            knowledge = item.get("instruction", "")
            code = item.get("output", "")
            long_term_memory.append({
                "question": question,
                "knowledge": knowledge,
                "code": code
            })
    else:
        for item in contents:
            question = item.get("template", "") or item.get("question", "")
            knowledge = ""  # You can adapt if EHR datasets have a "knowledge" field
            code = item.get("query", "") or ""
            long_term_memory.append({
                "question": question,
                "knowledge": knowledge,
                "code": code
            })
    # for item in contents:
    #     question = item.get("template", "")    # or item.get("question", "")
    #     knowledge = ""                         # or use another field if available
    #     code = item.get("query", "")           # or the solution field
    #     new_item = {
    #         "question": question,
    #         "knowledge": knowledge,
    #         "code": code
    #     }
    #     long_term_memory.append(new_item)

    # ---- NEW: Build embedding index for RAG ----
    questions_for_embed = [item["question"] for item in long_term_memory]
    user_proxy.set_embedding_index(questions_for_embed)
    user_proxy.update_memory(args.num_shots, long_term_memory)

    for i in range(args.start_id, args.num_questions):
        if args.debug and contents[i]['id'] != args.debug_id:
            continue
        # Select the correct field based on dataset
        if args.dataset.lower().startswith("health"):
            question = contents[i].get('input', '')
            answer = contents[i].get('output', '')
        else:
            question = contents[i].get('template', '') or contents[i].get('question', '')
            answer = contents[i].get('answer', '') or contents[i].get('query', '')

        try:
            # RAG retrieval for in-context demos
            examples = user_proxy.retrieve_examples_rag(question, k=args.num_shots)
            # If you use 'examples' in prompt construction, update agent prompt logic accordingly

            if chatbot is not None:
                user_proxy.initiate_chat(
                    chatbot,
                    message=question,
                )
                logs = user_proxy._oai_messages

                logs_string = []
                logs_string.append(str(question))
                logs_string.append(str(answer))
                for agent in list(logs.keys()):
                    for j in range(len(logs[agent])):
                        if logs[agent][j]['content'] is not None:
                            logs_string.append(logs[agent][j]['content'])
                        else:
                            argums = logs[agent][j]['function_call']['arguments']
                            if type(argums) == dict and 'cell' in argums.keys():
                                logs_string.append(argums['cell'])
                            else:
                                logs_string.append(argums)
            else:
                # Directly call your Hugging Face API handler here!
                config = config_list[0]
                prompt = question  # Optionally include 'examples' in prompt if you want few-shot
                try:
                    if config.get("api_type", "") == "LOCAL":
                        # Call local Llama-2
                        prediction = llama2_local_inference(prompt)
                        logs_string = [str(question), str(answer), prediction]
                    else:
                        # Call HF Inference API for other hosted models
                        prediction = hf_inference_api_call(prompt, config)
                        logs_string = [str(question), str(answer), prediction]
                except Exception as e:
                    logs_string = [str(e)]
        except Exception as e:
            logs_string = [str(e)]
        print(logs_string)
        # file_directory = file_path.format(id=contents[i]['id'])
        # Use the existing id if present, else fallback to index i
        item_id = contents[i].get('id', f'health_{i}')
        file_directory = file_path.format(id=item_id)
        os.makedirs(os.path.dirname(file_directory), exist_ok=True)

        if type(answer) == list:
            answer = ', '.join(answer)
        logs_string.append("Ground-Truth Answer ---> "+answer)

        with open(file_directory, 'w') as f:
            f.write('\n----------------------------------------------------------\n'.join(logs_string))
        logs_string = '\n----------------------------------------------------------\n'.join(logs_string)
        if '"cell": "' in logs_string:
            last_code_start = logs_string.rfind('"cell": "')
            last_code_end = logs_string.rfind('"\n}')
            last_code = logs_string[last_code_start+9:last_code_end]
        else:
            last_code_end = logs_string.rfind('Solution:')
        prediction_end = logs_string.rfind('TERMINATE')
        prediction = logs_string[last_code_end:prediction_end]
        result = judge(prediction, answer)
        if result:
            new_item = {"question": question, "knowledge": user_proxy.knowledge, "code": user_proxy.code}
            long_term_memory.append(new_item)
    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)

if __name__ == "__main__":
    main()
