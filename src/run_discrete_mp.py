import os
import torch.multiprocessing as mp
import time
import torch
import constants
import process
import parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def process_stream(dataloader, model):
    for step, batch in tqdm(enumerate(dataloader)):
        batch = {k: v.to(torch.device("cuda:0")) for k, v in batch.items()}  # Move data to GPU
        with torch.no_grad():
            generated = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], do_sample=False, max_new_tokens=constants.MAX_NEW_TOKENS)

if __name__ == "__main__":
    # parse arguments. set data paths based on expmt params
    args = parser.get_parser()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    torch.cuda.cudart().cudaProfilerStart()
    device = torch.device("cuda:0")
    
    mp.set_start_method('spawn', force=True)
    args.nproc = 2
    
    model = AutoModelForCausalLM.from_pretrained(constants.MODELS[args.model], return_dict=True, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(constants.MODELS[args.model], device_map=device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load data
    datasets = process.load_data(args, task='trn')
    batch_size = 20
    model.eval()

    # generate summary for each finding
    t0 = time.time()
    token_processes = []
    prompt_processes = []
    for rank in range(args.nproc):
        # Prompt Phase
        prompt_queue = mp.Queue()
        prompt_proc = mp.Process(target=process.get_loader, args=(datasets[rank], tokenizer, batch_size, prompt_queue))
        prompt_proc.start()
        prompt_processes.append(prompt_proc) 
        dataloader = prompt_queue.get()

        # Token Phase
        token_proc = mp.Process(target=process_stream, args=(dataloader, model))
        token_proc.start()
        token_processes.append(token_proc)

    prompt_time = time.time() - t0
    print('generated {} prompts for {} expmt in {} sec'.format(len(datasets[0]), args.expmt_name, prompt_time))
    for prompt_proc, token_proc in zip(prompt_processes, token_processes):
        prompt_proc.join()
        token_proc.join()

    end_time = time.time() - t0
    torch.cuda.cudart().cudaProfilerStop()
    print('generated {} samples for {} expmt in {} sec'.format(len(datasets[0]), args.expmt_name, end_time))