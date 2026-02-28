import pandas as pd
import os, sys
import numpy as np
from tqdm import tqdm

import argparse
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
import torch
import gc
from peft import LoraConfig, get_peft_model, TaskType
import accelerate
import copy

# from pe.llm import sft_fine_tune, sft_fine_tune_until_converge, get_per_sample_loss
# TODO: install DPSDA
from pesgd.logging import setup_logging, execution_logger
# from pe.runner import PE
from pe.dp import Gaussian
from pe.api.text import LLMAugPE
# from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import ComputeFID, ComputePrecisionRecall
from pe.callback import SaveTextToCSV
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

from pesgd.data import Data
from pesgd.data.text import TextCSV
from pesgd.embedding.text import SentenceTransformer
from pesgd.population import PESGDPopulation
from pesgd.llm import HuggingfaceLLM, NONE_INSTRUCT_MODELS
from pesgd.llm.fine_tune.trainer import per_sample_loss_function
from pesgd.utils.batch_memory_manager import BatchMemoryManager
from pesgd.data.data_utils import prepared_dataloader, get_initial_none_priv_data, get_num_samples_per_label_id, log_metrics
from pesgd.privacy_engine import PrivacyEngine
from pesgd.llm.fine_tune.fine_tuned_model_eval import evaluate_model_on_private_data

LOGGING_INTERVAL = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, metavar='seed', help='random seed (default: 0)')
    parser.add_argument('--gpu', default=0, type=int, help='gpu device id')
    parser.add_argument("--task", type=str, default="biorxiv", help="The name of the task.")
    parser.add_argument("--gen_model_name", type=str, default="openai-community/gpt2", help="The name of the model for generation.")
    parser.add_argument("--fine_tune_model_name", type=str, default="openai-community/gpt2", help="The name of the model for fine-tuning on private data.")
    parser.add_argument("--use_local_model", type=bool, default=False, help="Where to use huggingface automatic download or use local prepared models.")
    
    parser.add_argument("--fine_tune_model_train_iter", type=int, default=2, help="Numbers of iterations for training the target model for SFT. Default 2.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size. Default as 8.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate. Default as 5e-5.")
    parser.add_argument("--lr_scaler", type=float, default=0.5, help="Learning rate scaler for grid search.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank, default 8.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha, default 32.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate, default 0.1")
    parser.add_argument("--target_modules", default=['q_proj', 'v_proj'], nargs='+', type=str)

    parser.add_argument("--variation_api_fold", type=int, default=2, help="How many variations to apply to the initial synthetic data. Default 2, which means N*(2+1) samples will be generated.")
    parser.add_argument("--num_synthetic_per_iter", type=int, default=1000, help="Numbers of synthetic samples to generate per iteration (before api fold). Default 1000.")
    parser.add_argument("--num_iter", type=int, default=10, help="Numbers of iterations for synthetic samples. Default 10.")
    parser.add_argument("--priv_sampling_ratio", type=float, default=1.0, help="Numbers of synthetic samples to generate per iteration (before api fold). Default 1.0")
    parser.add_argument("--max_completion_tokens", type=int, default=1024, help="The maximum number of tokens to generate in the response. Should be related to task.")
    # parser.add_argument("--dp_mechanism", type=str, default='Gaussian', help="['Exponential', 'Gaussian'] The differential privacy mechanism to use.")
    parser.add_argument("--dp_epsilon", type=float, default=1.0, help="[1,2,4,inf(>=100000000.0)], (epsilon, delta)-DP protection of the histogram for each party, delata=0.0 for exponential mechanism")
    parser.add_argument("--dp_delta", type=float, default=0.0, help="[0.0, 1E-5, ...], (epsilon, delta)-DP protection of the histogram for each party")
    parser.add_argument("--dp_clip_norm", type=float, default=1.0, help="Clipping threshold for DP noise addition")
    parser.add_argument("--metric_inverse_epsilon", type=float, default=1e-6, help="The small epsilon used to inverse GTG. Default 1e-6.")
    
    parser.add_argument("--memory_check", type=bool, default=False, help="Whether to enable memory check during execution for debug.")


    args = parser.parse_args()

    # ########### initialize random seed ############
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # ########### initialize random seed ############


    # ########### load the private dataset for training and evaluation ############
    root_dir=f"./data/{args.task}"
    # priv_train_data = TextCSV(csv_path=os.path.join(root_dir, f"train.csv"), num_samples=40, rand_seed=args.seed)
    priv_train_data = TextCSV(csv_path=os.path.join(root_dir, f"train.csv"), num_samples=400, rand_seed=args.seed)
    # priv_train_data = TextCSV(csv_path=os.path.join(root_dir, f"train.csv"), num_samples=1000, rand_seed=args.seed)
    # priv_train_data = TextCSV(csv_path=os.path.join(root_dir, f"train.csv"), num_samples=4000, rand_seed=args.seed)
    # priv_train_data = TextCSV(csv_path=os.path.join(root_dir, f"train.csv"), num_samples=None, rand_seed=args.seed) # use all the samples
    # priv_test_data = TextCSV(csv_path=os.path.join(root_dir, f"test.csv"), num_samples=40, rand_seed=args.seed)
    # priv_test_data = TextCSV(csv_path=os.path.join(root_dir, f"test.csv"), num_samples=400, rand_seed=args.seed)
    priv_test_data = TextCSV(csv_path=os.path.join(root_dir, f"test.csv"), num_samples=None, rand_seed=args.seed) # use all the samples
    # print(f"[debug] Loaded private training data with {len(priv_train_data.data_frame)} samples.")
    # print(f"[debug] Loaded private testing data with {len(priv_test_data.data_frame)} samples.")
    # ########### load the private dataset for training and evaluation ############


    # ########### experiment folder and none_priv_data logging folder ############
    # exp_folder = f"./results_new{'_debug' if (args.debug!=0) else ''}/{args.method}{_prompt_selection}{_with_instruction_base}/{args.setting}/text/{args.task}/{args.dp_mechanism}_{args.dp_epsilon}_{args.dp_delta}_{args.variation_api_fold}fold_priv{len(data.data_frame)}/{args.gen_model_name.split('/')[-1]}/{args.fine_tune_model_name.split('/')[-1]}/{args.selection_model_name.split('/')[-1]}/[{args.num_synthetic_per_iter}]_{args.num_iter}_select{args.selection_model_train_iter}_finetune{args.fine_tune_model_train_iter}_priveratio{args.priv_sampling_ratio}_lr{args.lr}_MetInvEps{args.metric_inverse_epsilon}_clipNorm{args.dp_clip_norm}/seed{args.seed}/"
    exp_folder = f"./results_debug/text/{args.task}/{args.dp_epsilon}_{args.dp_delta}_{args.variation_api_fold}fold_priv{len(priv_train_data.data_frame)}/{args.gen_model_name.split('/')[-1]}/{args.fine_tune_model_name.split('/')[-1]}/[{args.num_synthetic_per_iter}]_{args.num_iter}_finetune{args.fine_tune_model_train_iter}_priveratio{args.priv_sampling_ratio}_lr{args.lr}_MetInvEps{args.metric_inverse_epsilon}_clipNorm{args.dp_clip_norm}/seed{args.seed}/"
    os.makedirs(exp_folder, exist_ok=True)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))
    
    # init_data_folder = f"./results_new{'_debug' if (args.debug!=0) else ''}/_initial_data/text/{args.task}/{args.variation_api_fold}fold/{args.gen_model_name.split('/')[-1]}/[{args.num_synthetic_per_iter}]/seed{args.seed}/synthetic_text/"
    init_data_folder = f"./results_debug/_initial_data/text/{args.task}/{args.variation_api_fold}fold/{args.gen_model_name.split('/')[-1]}/[{args.num_synthetic_per_iter}]/seed{args.seed}/synthetic_text/"
    init_data_checkpoint = f"{init_data_folder}000000000/"
    os.makedirs(init_data_checkpoint, exist_ok=True)
    # ########### experiment folder and none_priv_data logging folder ############


    # ########### load LLM for generation and fine-tuning ############
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO: is this neccessary?
    args.llm_add_instruction = (args.fine_tune_model_name not in NONE_INSTRUCT_MODELS)
    print(f"{args.llm_add_instruction=}")
    # args.llm_add_instruction = True
    glm = HuggingfaceLLM(max_completion_tokens=args.max_completion_tokens, model_name_or_path=args.gen_model_name, temperature=1.0, device_map='auto', gen_with_instruction=args.llm_add_instruction, use_local_model=args.use_local_model) # generation model
    if args.memory_check:
        print(f"[debugging] after loading generation model, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    llm = HuggingfaceLLM(max_completion_tokens=args.max_completion_tokens, model_name_or_path=args.fine_tune_model_name, temperature=1.0, device_map=None, gen_with_instruction=args.llm_add_instruction, use_local_model=args.use_local_model) # fine-tune model
    accelerator = accelerate.Accelerator()
    llm_lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM, 
    )
    llm._model = get_peft_model(llm._model, llm_lora_config)
    llm._model = accelerator.prepare(llm._model)
    llm._model, llm_offload_hook = accelerate.cpu_offload_with_hook(llm._model, execution_device="cuda")
    llm_offload_hook.offload()
    if args.memory_check:
        print(f"[debugging] after loading target training model, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # for _, module in llm._model.named_modules():
    #     if "lora" in module.__class__.__name__.lower():
    #         module.to(dtype=torch.float16, device=next(llm._model.parameters()).device)
    # for name, param in llm._model.named_parameters():
        print(f"{name}: shape={param.shape}, values={param.data.flatten()[:50]}")
    print(f"{llm._model.dtype=}, {llm._model.config=}")
    # ########### load LLM for generation and fine-tuning ############


    # ########### Data Evolution API ############
    api = LLMAugPE(
        llm=glm,
        random_api_prompt_file=os.path.join(current_folder, f"prompt/{args.task}/random_api_prompt.json"),
        variation_api_prompt_file=os.path.join(current_folder, f"prompt/{args.task}/variation_api_prompt.json"),
    )
    embedding = SentenceTransformer(model="sentence-t5-base", use_local_model=args.use_local_model)
    population = PESGDPopulation(
        api=api, initial_variation_api_fold=args.variation_api_fold, next_variation_api_fold=args.variation_api_fold, keep_selected=True, selection_mode="random", # selection_mode="random" for gradient version of PE
    )
    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid = ComputeFID(
        priv_data=priv_train_data, embedding=embedding, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1}
    )
    compute_precision_recall = ComputePrecisionRecall(
        priv_data=priv_train_data,
        embedding=embedding,
        filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1},
        num_precision_neighbors=10, # default is 4
        num_recall_neighbors=10, # default is 5
    )
    # compute_format_match = ComputeFormatMatch(
    #     format_type=args.task,
    # )
    save_text_to_csv = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()
    # ########### Data Evolution API ############


    # ########### prepare the private dataloader ############
    priv_train_dataset, priv_train_dataloader = prepared_dataloader(
        priv_train_data, llm._tokenizer, split_ratio=0.0,
        task_type="text", batch_size=int(len(priv_train_data.data_frame)*args.priv_sampling_ratio), shuffle=True,
        add_instruction=args.llm_add_instruction, instruction=population._api._random_api_prompt_config,
    )
    priv_test_dataset, priv_test_dataloader = prepared_dataloader(
        priv_test_data, llm._tokenizer, split_ratio=0.0,
        task_type="text", batch_size=args.train_batch_size*2, shuffle=False,
        add_instruction=args.llm_add_instruction, instruction=population._api._random_api_prompt_config,
    )
    optimizer = torch.optim.AdamW(llm._model.parameters(), lr=args.lr, weight_decay=0.01)
    args.num_classes = len(priv_train_data.metadata.label_info)
    # print(f"[debug] {args.num_classes=}")
    # ########### prepare the private dataloader ############


    # ########### prepare the synthetic dataloader ############
    none_priv_data = get_initial_none_priv_data(
        data_metadata=priv_train_data.metadata, 
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
        init_data_file=init_data_checkpoint,
        num_samples_schedule=[args.num_synthetic_per_iter] * (args.num_iter+1), 
        fraction_per_label_id=[1/args.num_classes]*args.num_classes,
        population=population,
        callbacks=[save_checkpoints, save_text_to_csv, compute_fid, compute_precision_recall],
        loggers=[csv_print, log_print],
    )
    # ########### prepare the synthetic dataloader ############

    # ########### base model evaluate ############
    eval_loss, eval_accuracy = evaluate_model_on_private_data(
        model=llm._model, tokenizer=llm._tokenizer, dataloader=priv_test_dataloader
    )
    print(f"Init Model | Eval loss: {eval_loss:.4f} | Eval accuracy: {eval_accuracy:.4f} | ɛ: 0.00")
    execution_logger.info(f"Init Model | Eval loss: {eval_loss:.4f} | Eval accuracy: {eval_accuracy:.4f} | ɛ: 0.00")
    if args.memory_check:
        print(f"[debugging] after evaluating the initial model, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # ########### base model evaluate ############

    # ########### base model SFT ############
    # sft on llm._model with all the none private data for several epochs, to get a better initial model for PE-SGD
    none_priv_train_dataset, none_priv_train_dataloader = prepared_dataloader(
        none_priv_data, llm._tokenizer, split_ratio=0.0,
        task_type="text", batch_size=args.train_batch_size, shuffle=False,
        add_instruction=args.llm_add_instruction, instruction=population._api._random_api_prompt_config,
    )
    normal_optimizer = torch.optim.AdamW(llm._model.parameters(), lr=args.lr, weight_decay=0.01)
    for epoch in range(1, args.fine_tune_model_train_iter+1):
        llm._model.train()
        for none_priv_batch in none_priv_train_dataloader:
            normal_optimizer.zero_grad()
            # print(f"[debug] in file <main_text> none private gradient gathering: {none_priv_batch=}")
            none_priv_batch = {key: val.to(device) for key, val in none_priv_batch.items()}
            # print(f"[debug] in file <main_text> none private gradient gathering after moving to device: {none_priv_batch=}")
            none_priv_inputs = {'input_ids':  none_priv_batch['input_ids'],
                            'attention_mask': none_priv_batch['attention_mask'],
                            # 'token_type_ids': none_priv_batch[2],
                            'labels':         none_priv_batch['labels']}
            none_priv_outputs = llm._model(**none_priv_inputs) # output = loss, logits, hidden_states, attentions
            none_priv_loss = per_sample_loss_function(none_priv_inputs['labels'], none_priv_outputs.logits, shift_label=True, reduction='none').sum()
            none_priv_loss.backward()
            normal_optimizer.step()
    eval_loss, eval_accuracy = evaluate_model_on_private_data(
        model=llm._model, tokenizer=llm._tokenizer, dataloader=priv_test_dataloader
    )
    print(f"SFTed-base Model | Eval loss: {eval_loss:.4f} | Eval accuracy: {eval_accuracy:.4f} | ɛ: 0.00")
    execution_logger.info(f"SFTed-base Model | Eval loss: {eval_loss:.4f} | Eval accuracy: {eval_accuracy:.4f} | ɛ: 0.00")
    torch.cuda.empty_cache()
    gc.collect()
    llm._model.train()
    # ########### base model SFT ############

    # ########### PE-SGD privatize the model ############
    if not 'ImageProcessor' in llm._tokenizer.__class__.__name__:
        llm._model.enable_input_require_grads()
    privacy_engine = PrivacyEngine()
    llm._model, optimizer, _priv_train_dataloader = privacy_engine.make_private_with_epsilon(
        module=llm._model,
        optimizer=optimizer,
        data_loader=priv_train_dataloader,
        target_delta=args.dp_delta,
        target_epsilon=args.dp_epsilon,
        # sampling_ratio=args.priv_sampling_ratio,
        epochs=args.num_iter,
        max_grad_norm=args.dp_clip_norm,
        poisson_sampling=True, # as default, set poisson sampling as True
        loss_reduction='sum', # we need per-sample gradients, although we do not used the built-in loss calculation, we make it clear here
        metric_inverse_epsilon=args.metric_inverse_epsilon,
        noise_on_vote=True, use_eigen=False, clip_or_normalize='normalize',
    )
    llm._model, llm_offload_hook = accelerate.cpu_offload_with_hook(llm._model, execution_device="cuda")
    # llm_offload_hook.offload()
    # ########### PE-SGD privatize the model ############
    if args.memory_check:
        print(f"[debugging] after all preparations are done, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")


    # ########### train for 1 step ############
    # print(f"[debug] in file <main_text>, current working device: {next(llm._model.parameters()).device=}")
    # print(f"[debug] in file <main_text>, expected device: {device=}")
    for epoch in range(1, args.num_iter+1):
        llm._model.train()
        # print(f"Epoch [{epoch}], {type(llm._model)=}, {type(llm._model._module)=}, {type(glm._model)=}")
        losses = []

        none_priv_train_dataset, none_priv_train_dataloader = prepared_dataloader(
            none_priv_data, llm._tokenizer, split_ratio=0.0,
            task_type="text", batch_size=args.train_batch_size, shuffle=False,
            add_instruction=args.llm_add_instruction, instruction=population._api._random_api_prompt_config,
        )

        print(f"[INFO] Epoch {epoch} none private gradient gathering ...")
        execution_logger.info(f"Epoch {epoch} none private gradient gathering ...")
        optimizer.zero_grad()
        _original_grad_accumulation_allowance = llm._model.grad_accumulation_allowed
        llm._model.allow_grad_accumulation()
        for none_priv_batch in none_priv_train_dataloader:
            # print(f"[debug] in file <main_text> none private gradient gathering: {none_priv_batch=}")
            none_priv_batch = {key: val.to(device) for key, val in none_priv_batch.items()}
            # print(f"[debug] in file <main_text> none private gradient gathering after moving to device: {none_priv_batch=}")
            none_priv_inputs = {'input_ids':  none_priv_batch['input_ids'],
                            'attention_mask': none_priv_batch['attention_mask'],
                            # 'token_type_ids': none_priv_batch[2],
                            'labels':         none_priv_batch['labels']}
            none_priv_outputs = llm._model(**none_priv_inputs) # output = loss, logits, hidden_states, attentions
            none_priv_loss = per_sample_loss_function(none_priv_inputs['labels'], none_priv_outputs.logits, shift_label=True, reduction='none').sum()
            none_priv_loss.backward()
        print(f"[INFO] Epoch {epoch} none private gradient storing ...")
        execution_logger.info(f"Epoch {epoch} none private gradient storing ...")
        optimizer.save_none_priv_gradients()
        del none_priv_loss, none_priv_outputs, none_priv_inputs, none_priv_batch
        torch.cuda.empty_cache()
        gc.collect()
        if not _original_grad_accumulation_allowance:
            llm._model.forbid_grad_accumulation()
        assert _original_grad_accumulation_allowance == llm._model.grad_accumulation_allowed, "[Error] Grad accumulation allowance state not properly restored."
        print(f"[INFO] Epoch {epoch} none private gradient stored.")
        execution_logger.info(f"Epoch {epoch} none private gradient stored.")
        if args.memory_check:
            print(f"[debugging] after none private gradient calculation and storage, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

        priv_batch_data = priv_train_data.poisson_sampling(sampling_ratio=args.priv_sampling_ratio)
        # print(f"[debug] in file <main_text> private gradient gathering of one sampling_ratio_batch, {input_ids.shape=}, {attention_mask.shape=}, {labels.shape=}")        
        priv_batch_dataset, dataloader = prepared_dataloader(
            priv_batch_data, llm._tokenizer, split_ratio=0.0,
            task_type="text", batch_size=args.train_batch_size, shuffle=True,
            add_instruction=args.llm_add_instruction, instruction=population._api._random_api_prompt_config,
        )
        # print(f"[debug] in file <main_text> private gradient gathering of one sampling_ratio_batch, {len(priv_batch_dataset)=}")
        optimizer.zero_grad()
        _original_grad_accumulation_allowance = llm._model.grad_accumulation_allowed
        llm._model.allow_grad_accumulation()
        for step, priv_batch in enumerate(tqdm(dataloader)):
            priv_batch = {key: val.to(device) for key, val in priv_batch.items()}
            # print(f"[debug] in file <main_text> none private gradient gathering after moving to device: {priv_batch=}")
            priv_inputs = {'input_ids':  priv_batch['input_ids'],
                        'attention_mask': priv_batch['attention_mask'],
                        # 'token_type_ids': priv_batch[2],
                        'labels':         priv_batch['labels']}

            priv_outputs = llm._model(**priv_inputs) # output = loss, logits, hidden_states, attentions

            # loss = outputs[0]
            # print(f"[debug] in file <main_text> private gradient gathering: {priv_inputs['labels'].device=}, {outputs.logits.device=}")
            priv_loss = per_sample_loss_function(priv_inputs['labels'], priv_outputs.logits, shift_label=True, reduction='none').sum()
            priv_loss.backward()
            losses.append(priv_loss.item())
        print(f"[INFO] Epoch {epoch} private gradient calculation done.")
        execution_logger.info(f"Epoch {epoch} private gradient calculation done.")
        if args.memory_check:
            print(f"[debugging] after private gradient calculation, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        optimizer.step() # get all priv_sample_grad and perform gradient projection and update the model accordingly
        privacy_engine.accountant.step(noise_multiplier=optimizer.noise_multiplier, sample_rate=args.priv_sampling_ratio)
        del priv_loss, priv_outputs, priv_inputs, priv_batch
        torch.cuda.empty_cache()
        gc.collect()
        if not _original_grad_accumulation_allowance:
            llm._model.forbid_grad_accumulation()
        assert _original_grad_accumulation_allowance == llm._model.grad_accumulation_allowed, "[Error] Grad accumulation allowance state not properly restored."
        train_loss = np.mean(losses)
        if args.memory_check:
            print(f"[debugging] after model update, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        
        # print(f"[debugging] in <main_text.py> to see where the privacy_engin is created. point-1")
        eps = privacy_engine.get_epsilon(args.dp_delta)
        # print(f"[debugging] in <main_text.py> to see where the privacy_engin is created. point-2")
        eval_loss, eval_accuracy = evaluate_model_on_private_data(
            model=llm._model, tokenizer=llm._tokenizer, dataloader=priv_test_dataloader
        )
        print(f"Epoch: {epoch} | Step: {step} | Train loss: {train_loss:.4f} | Eval loss: {eval_loss:.4f} | Eval accuracy: {eval_accuracy:.4f} | ɛ: {eps:.2f}")
        execution_logger.info(f"Epoch: {epoch} | Step: {step} | Train loss: {train_loss:.4f} | Eval loss: {eval_loss:.4f} | Eval accuracy: {eval_accuracy:.4f} | ɛ: {eps:.2f}")
    
        # ############# select sample for next epoch generation #############
        G_coefficient = optimizer.get_G_coefficient()
        print(f"Final G coefficient: {G_coefficient}")
        execution_logger.info(f"Final G coefficient: {G_coefficient}")
        train_sample_dp_score = torch.nn.functional.softmax(torch.abs(G_coefficient)).cpu().numpy()
        # print(f"[debugging] [epoch {epoch}] softmax(abs) train_sample_dp_score: {train_sample_dp_score=}, {len(train_sample_dp_score)=}")
        execution_logger.info(f"[epoch {epoch}] softmax(abs) train_sample_dp_score: {train_sample_dp_score=}, {len(train_sample_dp_score)=}")
        selected_sample_indices = np.random.choice(range(len(train_sample_dp_score)), size=args.num_synthetic_per_iter, p=train_sample_dp_score, replace=False)
        # print(f"[debugging] [epoch {epoch}] softmax(abs) selected_sample_indices: {selected_sample_indices=}, {len(selected_sample_indices)=}")
        execution_logger.info(f"[epoch {epoch}] softmax(abs) selected_sample_indices: {selected_sample_indices=}, {len(selected_sample_indices)=}")

        fraction_per_label_id = [1/args.num_classes] * args.num_classes # TODO: see if we add another way
        num_samples_per_label_id = get_num_samples_per_label_id(
            data_metadata=priv_train_data.metadata, # assume the number of label of private data is public knowledge
            num_samples=args.num_synthetic_per_iter,
            fraction_per_label_id=fraction_per_label_id,
        )

        # Generate synthetic data for each label.
        syn_data_list = []
        selected_syn_data = none_priv_data.select_by_index(selected_sample_indices)
        for label_id, label_info in enumerate(priv_train_data.metadata.label_info):
            execution_logger.info(f"Label {label_id}")
            sub_syn_data = selected_syn_data.filter_label_id(label_id=label_id)
            # print(f"[debugging] checking sub_syn_data len for {label_id=}: {len(sub_syn_data.data_frame)=}")
            # print(f"[debugging] before variation api in {epoch=} for label {label_id}, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

            if population._initial_variation_api_fold > 0:
                # Generate next population.
                sub_syn_data = population.next(
                    syn_data=sub_syn_data,
                    num_samples=len(sub_syn_data.data_frame), # for each label, next generation will generate self._population._initial_variation_api_fold variantions as duplications
                    selected=True,
                )
            else:
                # print(f"[debugging] no variation api for {epoch=}, just randomly generate new samples")
                sub_syn_data = population.initial(
                    label_info=label_info,
                    num_samples=num_samples_per_label_id[label_id],
                )
            sub_syn_data.set_label_id(label_id)
            syn_data_list.append(sub_syn_data)

            # print(f"[debugging] before variation api in {epoch=} for label {label_id}, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
            
        # print(f"[debugging] in <./examples/main_text.py> {selected_syn_data=}, {type(selected_syn_data)=}")
        if not population._keep_selected:
            none_priv_data = Data.concat([selected_syn_data] + syn_data_list, metadata=priv_train_data.metadata)
        else:
            none_priv_data = Data.concat(syn_data_list, metadata=priv_train_data.metadata)
        none_priv_data.data_frame.reset_index(drop=True, inplace=True)
        none_priv_data.metadata.iteration = epoch
        # print(f"[debugging] syn_data after concat: {len(none_priv_data.data_frame)=} with {none_priv_data.metadata.iteration=}")
        
        if args.memory_check:
            print(f"[debugging] after new sample_generation, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

        none_priv_data.save_checkpoint(os.path.join(exp_folder, "checkpoint"))
        log_metrics(
            syn_data=none_priv_data, 
            callbacks=[save_checkpoints, save_text_to_csv, compute_fid, compute_precision_recall],
            loggers=[csv_print, log_print]
        )
        # self._log_metrics(none_priv_data)
# finally:
#     self._clean_up_loggers()
        # ############# select sample for next epoch generation #############

        if args.gen_model_name == args.fine_tune_model_name:
            with torch.no_grad():
                if type(glm._model) == type(llm._model._module):
                    glm._model = copy.deepcopy(llm._model._module)
                else:
                    merged_model = copy.deepcopy(llm._model._module).merge_and_unload()
                    glm._model.load_state_dict(merged_model.state_dict(), strict=False)
                # print(f"[debugging] after updating generation model with fine-tuned model, check memory: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # ########### train for 1 step ############

    
    # ########### PE-SGD privatize the model ############
