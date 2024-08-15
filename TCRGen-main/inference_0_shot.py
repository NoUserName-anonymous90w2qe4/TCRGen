import os
import warnings
import csv
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
MODEL_DIR = "./models/TCRGen-0"
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

## Load novel epitopes that of interest
with open('./results/original/ood/epitope.txt', 'r') as file:
    epitopes = [line.strip() for line in file if line.strip()]


k_shots = 0
for EPITOPE in epitopes:
    EPITOPE_PROMPT = EPITOPE + '$'
    
    outputs = []
    for _ in tqdm(range(num_batches)):
        output = text_generator(EPITOPE_PROMPT, max_length=max_length_param, do_sample=do_sample_param, 
                                   top_k=top_k_param, repetition_penalty=repetition_penalty_param,
                                   num_return_sequences=batch_size, eos_token_id=eos_token_id_param, temperature=temperature)
        outputs.extend(output)


    print('Saving Epitope-TCR pairs into a csv file...')
    with open(f'./results/original/ood_temp_0.4/designed_TCRs/{EPITOPE}_{k_shots}_shots.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epi", "tcr"])

        for output in outputs:
            # Replace spaces and split by '$'
            split_text = output["generated_text"].replace(' ', '').split('$')

            # Get the epitope and the (k_shots+1)th TCR sequence
            epi = split_text[0]
            tcr = split_text[k_shots+1] if len(split_text) > 1 else "AA"  # Default to "AA" if no TCR is found

            if len(tcr) <= 1:
                tcr = "AA" 

            writer.writerow([epi, tcr])