import os
import warnings
import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from transformers import pipeline
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1' 
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


## Load your trained TCRGen
MODEL_DIR = "./models/TCRGen-k"
model = AutoModelFoCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./../RITA/RITA_m")

special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>'}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))  
tokenizer.pad_token = tokenizer.eos_token

## Use GPU for inference if possible
device_num = 0
device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")
model.to(device)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

## Inference Paramters
max_length_param = 64*3
do_sample_param = True
top_k_param = 8
repetition_penalty_param = 1.2
temperature = 0.4
eos_token_id_param = 2
batch_size = 50  
num_batches = 20  


## Helper function to sample known binding TCRs and prepare prompt
## used in k-shot inferences.
def preselect_tcrs_for_epitopes(file_path, epitopes, max_shots):
    """
    Preselect a fixed number of TCRs for each epitope, removing any <EOS> tokens.
    """
    epitope_tcr_map = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().replace('<EOS>', '').split('$')
            if len(parts) < 2:
                continue

            epi, tcr = parts[0], parts[1]
            if epi in epitopes:
                if epi not in epitope_tcr_map:
                    epitope_tcr_map[epi] = []
                if len(epitope_tcr_map[epi]) < max_shots:
                    epitope_tcr_map[epi].append(tcr)

    return epitope_tcr_map


def generate_k_shot_prompt(epitope, tcrs, k_shots, max_length=64, top_k=8, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=2, verbose=1):
    """
    Generate a prompt for k-shot inference using a subset of preselected TCRs.
    """
    selected_tcrs = tcrs[:k_shots]
    prompt = f"{epitope}${'$'.join(selected_tcrs)}$"

    # Generate sequences
    outputs = text_generator(prompt, max_length=max_length*k_shots, do_sample=True, top_k=top_k, repetition_penalty=repetition_penalty,
                            num_return_sequences=num_return_sequences, eos_token_id=eos_token_id,
                            temperature=temperature)

    res = []
    print(f"{k_shots}-shots Prompt:")
    for output in outputs:
        generated_text = output["generated_text"].replace(' ', '')

        if verbose == 0:
            split_text = generated_text.split('$')
            # Ensure there are enough elements after splitting and get the first generated TCR
            if len(split_text) > k_shots + 1:
                tcr = split_text[k_shots + 1]
            else:
                tcr = "AA"  # Default TCR if no valid TCR is found
            res.append(tcr)
        else:
            print(generated_text + '\n')
    return res



## Load novel epitopes that of interest
with open('./results/original/ood/epitope.txt', 'r') as file:
    epitopes = [line.strip() for line in file if line.strip()]

# Preselect TCRs
file_path = './../data/contains_greaterFreq_100_epis/epi_testing.txt'
max_shots = 5  # Maximum number of shots to consider
epitope_tcr_map = preselect_tcrs_for_epitopes(file_path, epitopes, max_shots)

# Generate and save TCRs in batches
for epi in tqdm(epitopes):
    for k_shots in range(1, 5):  
        
        tcrs = epitope_tcr_map.get(epi, [])
        num_sequences_per_batch = 100  # Set the number of sequences per batch
        total_sequences = 1000  # Total sequences you want
        num_batches = total_sequences // num_sequences_per_batch

        print('Saving Epitope-TCR pairs into a csv file...')
        with open(f'./results/original/k_shots_temp_0.4/designed_TCRs/{epi}_{k_shots}_shots.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epi", "tcr"])

            for batch in range(num_batches):
                tcrs = generate_k_shot_prompt(epi, tcrs, k_shots, num_return_sequences=num_sequences_per_batch, verbose=0)

                for tcr in tcrs:
                    if len(tcr) <= 1:
                        tcr = "AA"  # Replace too short sequences with placeholder
                    writer.writerow([epi, tcr])       