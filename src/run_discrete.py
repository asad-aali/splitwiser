import time
import torch
import constants
import process
import parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

if __name__ == "__main__":
    # parse arguments. set data paths based on expmt params
    args = parser.get_parser()
    torch.cuda.cudart().cudaProfilerStart()
    device = torch.device("cuda:0")
    
    model = AutoModelForCausalLM.from_pretrained(constants.MODELS[args.model], return_dict=True, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(constants.MODELS[args.model], device_map=device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load data
    args.nproc = 1
    test_dataset = process.load_data(args, task='trn')
    batch_size = 20
    t0 = time.time()
    test_loader = process.get_loader(test_dataset, tokenizer, batch_size, None)
    end_time = time.time() - t0
    print('generated {} prompts for {} expmt in {} sec'.format(len(test_dataset), args.expmt_name, end_time))
    model.eval()

    # generate summary for each finding
    t0 = time.time()
    for step, batch in tqdm(enumerate(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], do_sample=False, max_new_tokens=constants.MAX_NEW_TOKENS)
    end_time = time.time() - t0
    torch.cuda.cudart().cudaProfilerStop()
    print('generated {} samples for {} expmt in {} sec'.format(len(test_dataset), args.expmt_name, end_time))