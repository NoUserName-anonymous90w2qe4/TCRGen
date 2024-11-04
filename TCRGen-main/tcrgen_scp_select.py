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
os.environ["CUDA_VISIBLE_DEVICES"]='0' 
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


#### Load model checkpoints and tokenizer
MODEL_DIR = "./models/SFT/rita_m/10_shots" ## we use 5_shots as an example
# MODEL_DIR = "./models/SFT/rita_m_finetuned/5_shots/checkpoint-1200" ## we use 5_shots as an example
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./models/pLMs/RITA_m")

special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>'}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))  
tokenizer.pad_token = tokenizer.eos_token

#### load model to cuda
device_num = 0
device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")
model.to(device)

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

#### Paramters
EPITOPE = 'LPRRSGAAGA'
EPITOPE_PROMPT = EPITOPE + '$'

max_length_param = 64*5
do_sample_param = True
top_k_param = 8
repetition_penalty_param = 1.2
eos_token_id_param = 2
batch_size = 100  # Define a smaller batch size
num_batches = 1  # Use 20 to get 1000 sequences in total




#### load epitopes list that we are interested to generate TCRs for
with open('./data/epitopes.txt', 'r') as file:
    epitopes = [line.strip() for line in file if line.strip()]
    
#### TCR Generation with SCP select
# Set parameters here
k_shots = 9  # Maximum shots, from 1 to 4
temperature = 0.4
bap_threshold = 0.93  # default 0.47
gpt_threshold = 1.5  # default 1.06

# Loop over each epitope
# for EPITOPE in epitopes:
for EPITOPE in ['VLAWLYAAV', 'FLNRFTTTL', 'RPHERNGFTVL', 'TPINLVRDL']:       
    EPITOPE_PROMPT_BASE = EPITOPE + '$'
    
    # Load the corresponding TCR data for the current epitope
    # df_tcrs = pd.read_csv(f'./results/10_shots_ICT/scp_random/designed_TCRs/{EPITOPE}_0_tcrs_prompting.csv')
    df_tcrs = pd.read_csv(f'./results/10_shots_ICT/scp_select_1/designed_TCRs/{EPITOPE}_0_tcrs_prompting.csv')
    
    # Filter the TCRs based on bap_score and gpt_loglikelihood thresholds
    filtered_tcrs = df_tcrs[(df_tcrs['bap'] >= bap_threshold) & 
                            (df_tcrs['gpt_ll'] / df_tcrs['tcr'].str.len() >= gpt_threshold)]
    
    # Convert the filtered TCRs to a list
    selected_tcrs = filtered_tcrs['tcr'].to_list()

    # Loop for each shot level
    for shot in range(1, k_shots + 1):
        outputs = []
        
        # Build the prompt for the current number of shots
        prompt = EPITOPE_PROMPT_BASE
        for i in range(shot):
            if i < len(selected_tcrs):
                prompt += selected_tcrs[i] + '$'
            else:
                break  # Stop if there aren't enough selected TCRs
        
        # Generate outputs for the current prompt
        for _ in tqdm(range(num_batches)):
            output = text_generator(prompt, max_length=max_length_param, do_sample=do_sample_param,
                                    top_k=top_k_param, repetition_penalty=repetition_penalty_param,
                                    num_return_sequences=batch_size, eos_token_id=eos_token_id_param, temperature=temperature)
            outputs.extend(output)

        # Save outputs for the current shot level
        print(f'Saving Epitope-TCR pairs for shot {shot} into a csv file...')
        with open(f'./results/10_shots_ICT/scp_select_1/designed_TCRs/{EPITOPE}_{shot}_tcrs_prompting.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epi", "tcr"])

            for output in outputs:
                split_text = output["generated_text"].replace(' ', '').split('$')

                
                try:
                    # Get the epitope and the (k_shots+1)th TCR sequence
                    epi = split_text[0]
                    tcr = split_text[k_shots+1]
                except IndexError:
                    tcr = "AA"  # Default to "AA" if no TCR is found

                # Ensure tcr has at least two characters
                if len(tcr) <= 1:
                    tcr = "AA" 

                writer.writerow([epi, tcr])