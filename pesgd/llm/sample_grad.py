import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, DefaultDataCollator
from transformers import EarlyStoppingCallback
from typing import Optional
from datasets import Dataset
import accelerate
from peft import LoraConfig, get_peft_model, TaskType
from pesgd.grad_sample import GradSampleModule

import os
import json
import pandas as pd
import gc
import numpy as np
import random
import copy

from .fine_tune.metric_logging_plotting import logging_plotting
from .fine_tune.instruction_addition import chat_template_tokenize_example
from .fine_tune.image_dataset_processing import image_preprocess
from .fine_tune.fine_tuned_model_eval import evaluate_model_on_private_data
from .fine_tune.trainer import per_sample_loss_function
from .fine_tune.trainer_image import per_sample_loss_function_image
# from .ghostEngines.graddotprod_engine import GradDotProdEngine
from .ghostEngines.accumulate_graddotprod_engine import AccumulateGradDotProdEngine
from pe.logging import execution_logger


def opt_grad_combine(
    train_sample_grad,
    val_sample_grad,
    noise_multiplier=None,
    metric_inverse_epsilon=1e-6,
    clip_or_normalize='normalize',
    noise_on_vote=True,
    use_eigen=False,
    clip_norm=1.0,
):
    # Let G = train_sample_grad.T
    G = train_sample_grad.T  # shape: [#param, #train_sample]
    eps = metric_inverse_epsilon  # small epsilon for numerical stability
    GTG = G.T @ G  # shape: [#train_sample, #train_sample]
    GTG = GTG + eps * torch.eye(GTG.shape[0], device=GTG.device, dtype=GTG.dtype)  # add epsilon to diagonal
    # print(f"[debugging] see {G=}")
    # print(f"[debugging] see {val_sample_grad=}")

    if use_eigen == False:
        GTG_inv = torch.linalg.pinv(GTG)  # shape: [#train_sample, #train_sample]
        # # G_GTG_inv = G @ GTG_inv @ G.T # shape: [#param, #train_sample] * [#train_sample, #train_sample] * [#train_sample, #param] = [#param, #param]
        # G_GTG_inv = G @ GTG_inv # shape: [#param, #train_sample] * [#train_sample, #train_sample] = [#param, #train_sample]

        # store the none-noisy average val_sample_grad
        real_average_grad = copy.deepcopy(torch.mean(val_sample_grad, dim=0)) # inner_products.shape=[#val_sample, #param], real_average_grad.shape=[#param]

        if not noise_on_vote: 
            # add gaussian noise on private sample's clipped gradients: val_sample_grad with shape [#val_sample, #param]
            num_val = val_sample_grad.size(0)
            # step-1, clip and norm
            execution_logger.info(f"{val_sample_grad.norm(dim=1, keepdim=True)=}")
            print(f"val_sample_grad.norm mean={torch.mean(val_sample_grad.norm(dim=1, keepdim=True))}, std={torch.std(val_sample_grad.norm(dim=1, keepdim=True))}, min={torch.min(val_sample_grad.norm(dim=1, keepdim=True))}, max={torch.max(val_sample_grad.norm(dim=1, keepdim=True))}")
            execution_logger.info(f"val_sample_grad.norm mean={torch.mean(val_sample_grad.norm(dim=1, keepdim=True))}, std={torch.std(val_sample_grad.norm(dim=1, keepdim=True))}, min={torch.min(val_sample_grad.norm(dim=1, keepdim=True))}, max={torch.max(val_sample_grad.norm(dim=1, keepdim=True))}")
            # val_sample_grad = val_sample_grad / torch.clamp(val_sample_grad.norm(dim=1, keepdim=True) + 1e-8, min=clip_norm)  # [#val_sample, #param]
            val_sample_grad = val_sample_grad * (clip_norm / (val_sample_grad.norm(dim=1, keepdim=True) + 1e-8)).clamp(max=1.0)
            # print(f"[debugging] see clipped normalized {val_sample_grad=}")
            execution_logger.info(f"clipped normalized {val_sample_grad=}")
            # step-2 sum
            val_sample_grad = torch.sum(val_sample_grad, dim=0)  # [#param]
            # print(f"[debugging] see clipped normalized summed {val_sample_grad=}")
            execution_logger.info(f"clipped normalized summed {val_sample_grad=}")
            # step-3 add noise
            val_sample_grad += torch.tensor(np.random.normal(scale=noise_multiplier*clip_norm, size=val_sample_grad.size(0))).to(val_sample_grad.device)
            # print(f"[debugging] see clipped normalized summed and noised {val_sample_grad=}")
            execution_logger.info(f"clipped normalized summed and noised {val_sample_grad=}")
            # step-4 average
            val_sample_grad = val_sample_grad / num_val
            # print(f"[debugging] see averaged clipped normalized summed and noised {val_sample_grad=}")
            execution_logger.info(f"clipped averaged normalized summed and noised {val_sample_grad=}")

            real_noised_val_grad_cos_sim = torch.dot(real_average_grad, val_sample_grad) / (torch.norm(real_average_grad) * torch.norm(val_sample_grad))
            real_noised_val_grad_l2_distance = torch.norm(real_average_grad - val_sample_grad)
            # print(f"[debugging] {real_noised_val_grad_cos_sim=}, avg_real_grad_norm={torch.norm(real_average_grad)}, {real_noised_val_grad_l2_distance=}, avg_approximate_grad_norm={torch.norm(val_sample_grad)}, approximate/real norm ration={torch.norm(val_sample_grad)/torch.norm(real_average_grad)}")
            execution_logger.info(f"{real_noised_val_grad_cos_sim=}, avg_real_grad_norm={torch.norm(real_average_grad)}, {real_noised_val_grad_l2_distance=}, avg_approximate_grad_norm={torch.norm(val_sample_grad)}, approximate/real norm ration={torch.norm(val_sample_grad)/torch.norm(real_average_grad)}")

            # Inner product of each row of train_sample_grad and each row of val_sample_grad
            # Result shape: [#train_sample]
            inner_products = train_sample_grad @ val_sample_grad.view(-1).T

            # execution_logger.info(f"[debugging] {GTG.shape=}, {GTG=}")
            # execution_logger.info(f"[debugging] {GTG_inv.shape=}, {GTG_inv=}")
            # execution_logger.info(f"[debugging] {inner_products.shape=}, {inner_products=}")

            # G_coefficient for multiplying G
            # Result shape: [#train_sample]
            G_coefficient = GTG_inv @ inner_products
            # print(f"[debugging] see noised sample-grad resulted {G_coefficient=}")
            execution_logger.info(f"noised sample-grad resulted {G_coefficient=}")
            
            approximate_grad = G @ G_coefficient # shape: [#param]
            average_approximate_grad = approximate_grad
            print(f"{average_approximate_grad.shape=}")

        else: # add gaussian noise on private sample's summed up scores
            # Inner product of each row of train_sample_grad and each row of val_sample_grad
            # Result shape: [#train_sample, #val_sample]
            inner_products = train_sample_grad @ val_sample_grad.T

            # execution_logger.info(f"[debugging] {GTG.shape=}, {GTG=}")
            # execution_logger.info(f"[debugging] {GTG_inv.shape=}, {GTG_inv=}")
            # execution_logger.info(f"[debugging] {inner_products.shape=}, {inner_products=}")

            ################### [[[[dp on (G_GTG_inv @ inner_products)]]]] starts ###################
            # # final_grad: G_GTG_inv @ inner_products
            # # Result shape: [#param, #val_sample]
            # approximate_grad = G_GTG_inv @ inner_products

            # G_coefficient for multiplying G
            # Result shape: [#train_sample, #val_sample]
            G_coefficient = GTG_inv @ inner_products

            # add noise on to G_coefficient
            # print(f"[debugging] see non-normalized {G_coefficient=}")
            print(f"{G_coefficient.norm(dim=0, keepdim=True)=}")
            execution_logger.info(f"{G_coefficient.norm(dim=0, keepdim=True)=}")
            print(f"G_coefficient.norm mean={torch.mean(G_coefficient.norm(dim=0, keepdim=True))}, std={torch.std(G_coefficient.norm(dim=0, keepdim=True))}, min={torch.min(G_coefficient.norm(dim=0, keepdim=True))}, max={torch.max(G_coefficient.norm(dim=0, keepdim=True))}")
            execution_logger.info(f"G_coefficient.norm mean={torch.mean(G_coefficient.norm(dim=0, keepdim=True))}, std={torch.std(G_coefficient.norm(dim=0, keepdim=True))}, min={torch.min(G_coefficient.norm(dim=0, keepdim=True))}, max={torch.max(G_coefficient.norm(dim=0, keepdim=True))}")
            # asser 1 == 0
            if clip_or_normalize == 'clip':
                # G_coefficient = G_coefficient / torch.clamp(G_coefficient.norm(dim=0, keepdim=True) + 1e-8, min=clip_norm)
                G_coefficient = G_coefficient * (clip_norm / (G_coefficient.norm(dim=0, keepdim=True) + 1e-8)).clamp(max=1.0)
                # print(f"[debugging] see clipped normalized {G_coefficient=}")
                execution_logger.info(f"clipped normalized {G_coefficient=}")
                G_coefficient = torch.sum(G_coefficient, dim=-1) # should be of shape [#train_sample]
                # print(f"[debugging] see summed up {G_coefficient=}")
                execution_logger.info(f"summed up {G_coefficient=}")
                G_coefficient += torch.tensor(np.random.normal(scale=noise_multiplier*clip_norm, size=G_coefficient.size(0))).to(G_coefficient.device)
                # print(f"[debugging] see summed up and DP-ed {G_coefficient=}")
                execution_logger.info(f"summed up and DP-ed {G_coefficient=}")
            else:
                assert clip_or_normalize == 'normalize', f"[ERROR] only supporting normalize or clip for {clip_or_normalize=} currently"
                G_coefficient = G_coefficient / (G_coefficient.norm(dim=0, keepdim=True) + 1e-8)
                # print(f"[debugging] see normalized {G_coefficient=}")
                execution_logger.info(f"normalized {G_coefficient=}")
                G_coefficient = torch.sum(G_coefficient, dim=-1) # should be of shape [#train_sample]
                # print(f"[debugging] see summed up {G_coefficient=}")
                execution_logger.info(f"summed up {G_coefficient=}")
                G_coefficient += torch.tensor(np.random.normal(scale=noise_multiplier, size=G_coefficient.size(0))).to(G_coefficient.device)
                # print(f"[debugging] see summed up and DP-ed {G_coefficient=}")
                execution_logger.info(f"summed up and DP-ed {G_coefficient=}")
            ################### [[[[dp on (G_GTG_inv @ inner_products)]]]] ends ###################

            approximate_grad = G @ G_coefficient # shape: [#param]
            average_approximate_grad = approximate_grad / inner_products.size(-1) # shape: [#param]
            G_coefficient = G_coefficient / inner_products.size(-1)
            print(f"{average_approximate_grad.shape=}")
        
        del GTG
        del GTG_inv
        del G
        del inner_products
        # print(f"[debugging] check {train_sample_grad=} after 'del G'")
    else:
        num_syn = G.size(-1)
        # GTG = GTG / num_syn  # [#syn_sample, #syn_sample]
        eigvals, eigvecs = torch.linalg.eigh(GTG)   # eigvecs[:, i] is eigenvector for eigvals[i]
        # Sort in descending order (largest first)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Take the top 50 eigenvectors
        topk = 50
        U = eigvecs[:, :topk]   # shape [#param, 50]
        E = eigvals[:topk]
        # U is eigen vectors of GTG, but what we do need is GGT's eigen vectors V. 
        # Fortunately, we have V=GU with corresponding eigen values the same

        real_average_grad = copy.deepcopy(torch.mean(val_sample_grad, dim=0)) # inner_products.shape=[#val_sample, #param], real_average_grad.shape=[#param]

        num_val = val_sample_grad.size(0)
        # step-1, clip and norm
        # val_sample_grad = val_sample_grad / torch.clamp(val_sample_grad.norm(dim=1, keepdim=True) + 1e-8, min=clip_norm)  # [#val_sample, #param]
        val_sample_grad = val_sample_grad * (clip_norm / (val_sample_grad.norm(dim=1, keepdim=True) + 1e-8)).clamp(max=1.0)
        # print(f"[debugging] see clipped normalized {val_sample_grad=}")
        execution_logger.info(f"clipped normalized {val_sample_grad=}")
        # step-2 sum
        val_sample_grad = torch.sum(val_sample_grad, dim=0)  # [#param]
        # print(f"[debugging] see clipped normalized summed {val_sample_grad=}")
        execution_logger.info(f"clipped normalized summed {val_sample_grad=}")
        # step-3 add noise
        val_sample_grad += torch.tensor(np.random.normal(scale=noise_multiplier*clip_norm, size=val_sample_grad.size(0))).to(val_sample_grad.device)
        # print(f"[debugging] see clipped normalized summed and noised {val_sample_grad=}")
        execution_logger.info(f"clipped normalized summed and noised {val_sample_grad=}")
        # step-4 average
        val_sample_grad = val_sample_grad / num_val
        # print(f"[debugging] see averaged clipped normalized summed and noised {val_sample_grad=}")
        execution_logger.info(f"clipped averaged normalized summed and noised {val_sample_grad=}")

        real_noised_val_grad_cos_sim = torch.dot(real_average_grad, val_sample_grad) / (torch.norm(real_average_grad) * torch.norm(val_sample_grad))
        real_noised_val_grad_l2_distance = torch.norm(real_average_grad - val_sample_grad)
        # print(f"[debugging] {real_noised_val_grad_cos_sim=}, avg_real_grad_norm={torch.norm(real_average_grad)}, {real_noised_val_grad_l2_distance=}, avg_approximate_grad_norm={torch.norm(val_sample_grad)}, approximate/real norm ration={torch.norm(val_sample_grad)/torch.norm(real_average_grad)}")
        execution_logger.info(f"{real_noised_val_grad_cos_sim=}, avg_real_grad_norm={torch.norm(real_average_grad)}, {real_noised_val_grad_l2_distance=}, avg_approximate_grad_norm={torch.norm(val_sample_grad)}, approximate/real norm ration={torch.norm(val_sample_grad)/torch.norm(real_average_grad)}")

        # final grad = V @ V.T @ val_sample_grad
        #            = G @ U @ diag(E)**(-1) @ U.T @ G.T @ val_sample_grad; diag(E)**(-1) is to control the norm
        G_coefficient = ((U @ torch.diag(1/E) @ U.T) @ (G.T @ val_sample_grad))
        average_approximate_grad = G @ G_coefficient
        # print(f"[debugging] see eigened summed up and DP-ed {G_coefficient=}")
        execution_logger.info(f"summed eigened up and DP-ed {G_coefficient=}")
        print(f"{average_approximate_grad.shape=}")

        del GTG
        del G

    torch.cuda.empty_cache()
    gc.collect()

    return average_approximate_grad, G_coefficient, real_average_grad


def grad_projection(
    train_sample_grad,  # [#train_sample, #param]
    val_sample_grad,  # [#val_sample, #param]
    noise_multiplier=None,
    clip_or_normalize='clip', 
    with_residual=True,
    use_eigen=False,
):
    num_val = val_sample_grad.size(0)

    # Let G = train_sample_grad.T
    G = train_sample_grad.T  # shape: [#param, #train_sample]
    # H = val_sample_grad.T # shape: [#param, #val_sample]

    inner_products = train_sample_grad @ val_sample_grad.T  # shape: [#train_sample, #val_sample]
    residual = val_sample_grad.T - G @ inner_products  # shape: [#param, #val_sample]

    # store the none-noisy average val_sample_grad
    real_average_grad = copy.deepcopy(torch.mean(val_sample_grad, dim=0)) # val_sample_grad.shape=[#val_sample, #param], real_average_grad.shape=[#param]

    if with_residual:
        ################### [[[[dp on inner_products]]]] starts ###################
        #### always clip before dp, not normalization ####
        # inner_products = inner_products / torch.clamp(inner_products.norm(dim=0, keepdim=True) + 1e-8, min=1.0)
        inner_products = inner_products * (1.0 / (inner_products.norm(dim=0, keepdim=True) + 1e-8)).clamp(max=1.0)
        # print(f"[debugging] see clipped normalized {inner_products=}")
        execution_logger.info(f"clipped normalized {inner_products=}")
        # inner_products = inner_products / (inner_products.norm(dim=0, keepdim=True) + 1e-8)
        # print(f"[debugging] see normalized {inner_products=}")
        # execution_logger.info(f"normalized {inner_products=}")
        #### always clip before dp, not normalization ####
        inner_products = torch.sum(inner_products, dim=-1) # should be of shape [#train_sample]
        # print(f"[debugging] see summed up {inner_products=}")
        execution_logger.info(f"summed up {inner_products=}")
        inner_products += torch.tensor(np.random.normal(scale=noise_multiplier*1.0*np.sqrt(2), size=inner_products.size(0))).to(inner_products.device)
        # print(f"[debugging] see summed up and DP-ed {inner_products=}")
        execution_logger.info(f"summed up and DP-ed {inner_products=}")
        inner_products = inner_products / num_val
        ################### [[[[dp on inner_products]]]] ends ###################

        ################### [[[[dp on residual]]]] starts ###################
        #### always clip before dp, not normalization ####
        # residual = residual / torch.clamp(residual.norm(dim=0, keepdim=True) + 1e-8, min=0.2)  # shape: [#param, #val_sample]
        residual = residual * (0.2 / (residual.norm(dim=0, keepdim=True) + 1e-8)).clamp(max=1.0)
        # print(f"[debugging] see clipped normalized {residual=}")
        execution_logger.info(f"clipped normalized {residual=}")
        # residual = residual / (residual.norm(dim=0, keepdim=True) + 1e-8)
        # print(f"[debugging] see normalized {residual=}")
        # execution_logger.info(f"normalized {residual=}")
        #### always clip before dp, not normalization ####
        residual = torch.sum(residual, dim=-1) # should be of shape [#train_sample]
        # print(f"[debugging] see summed up {residual=}")
        execution_logger.info(f"summed up {residual=}")
        residual += torch.tensor(np.random.normal(scale=noise_multiplier*0.2*np.sqrt(2), size=residual.size(0))).to(residual.device)
        # print(f"[debugging] see summed up and DP-ed {residual=}")
        execution_logger.info(f"summed up and DP-ed {residual=}")
        residual = residual / num_val
        ################### [[[[dp on residual]]]] ends ###################
        average_approximate_grad = G @ inner_products + residual  # [#param, #train_sample] x [#train_sample, 1] + [#param] --> [#param]
    else:
        ################### [[[[dp on inner_products]]]] starts ###################
        #### always clip before dp, not normalization ####
        # inner_products = inner_products / torch.clamp(inner_products.norm(dim=0, keepdim=True) + 1e-8, min=1.0)
        inner_products = inner_products * (1.0 / (inner_products.norm(dim=0, keepdim=True) + 1e-8)).clamp(max=1.0)
        # print(f"[debugging] see clipped normalized {inner_products=}")
        execution_logger.info(f"clipped normalized {inner_products=}")
        # inner_products = inner_products / (inner_products.norm(dim=0, keepdim=True) + 1e-8)
        # print(f"[debugging] see normalized {inner_products=}")
        # execution_logger.info(f"normalized {inner_products=}")
        #### always clip before dp, not normalization ####
        inner_products = torch.sum(inner_products, dim=-1) # should be of shape [#train_sample]
        # print(f"[debugging] see summed up {inner_products=}")
        execution_logger.info(f"summed up {inner_products=}")
        inner_products += torch.tensor(np.random.normal(scale=noise_multiplier*1.0, size=inner_products.size(0))).to(inner_products.device)
        # print(f"[debugging] see summed up and DP-ed {inner_products=}")
        execution_logger.info(f"summed up and DP-ed {inner_products=}")
        inner_products = inner_products / num_val
        ################### [[[[dp on inner_products]]]] ends ###################

        average_approximate_grad = G @ inner_products

    del G
    # print(f"[debugging] check {train_sample_grad=} after 'del G'")
    torch.cuda.empty_cache()
    gc.collect()

    return average_approximate_grad, inner_products, real_average_grad


def gep(
    train_sample_grad,  # [#train_sample, #param]
    val_sample_grad,  # [#val_sample, #param]
    noise_multiplier=None,
    clip_or_normalize='clip', 
    with_residual=True,
    use_eigen=False,
):
    num_val = val_sample_grad.size(0)

    def orthogonalize(matrix):
        n, m = matrix.shape
        for i in range(m):
            # Normalize the i'th column
            col = matrix[:, i : i + 1]
            col /= torch.sqrt(torch.sum(col ** 2))
            # Project it on the rest and remove it
            if i + 1 < m:
                rest = matrix[:, i + 1 :]
                # rest -= torch.matmul(col.t(), rest) * col
                rest -= torch.sum(col * rest, dim=0) * col

    def check_approx_error(L, target):
        encode = torch.matmul(target, L) # n x k
        decode = torch.matmul(encode, L.T)
        error = torch.sum(torch.square(target - decode))
        target = torch.sum(torch.square(target))
        if(target.item()==0):
            return -1
        return error.item()/target.item()
        
    def get_bases(pub_grad, num_bases=50, power_iter=1):
        num_k = pub_grad.shape[0]
        num_p = pub_grad.shape[1]
    
        num_bases = min(num_bases, num_p)
        L = torch.normal(0, 1.0, size=(pub_grad.shape[1], num_bases), device=pub_grad.device)
        for i in range(power_iter):
            R = torch.matmul(pub_grad, L) # n x k
            L = torch.matmul(pub_grad.T, R) # p x k
            orthogonalize(L)
        error_rate = check_approx_error(L, pub_grad)
        return L, num_bases, error_rate

    train_sample_grad_basis, _, _error_rate = get_bases(train_sample_grad, num_bases=50, power_iter=1) # shape: [#param, 50]

    # # Let G = train_sample_grad.T
    # G = train_sample_grad.T  # shape: [#param, #train_sample]
    # # H = val_sample_grad.T # shape: [#param, #val_sample]
    G = train_sample_grad_basis # shape: [#param, 50=#basis=#eigenvec]

    inner_products = G.T @ val_sample_grad.T  # shape: [#train_sample, #val_sample]
    residual = val_sample_grad.T - G @ inner_products  # shape: [#param, #val_sample]

    # store the none-noisy average val_sample_grad
    real_average_grad = copy.deepcopy(torch.mean(val_sample_grad, dim=0)) # val_sample_grad.shape=[#val_sample, #param], real_average_grad.shape=[#param]

    if with_residual:
        ################### [[[[dp on inner_products]]]] starts ###################
        #### always clip before dp, not normalization ####
        # inner_products = inner_products / torch.clamp(inner_products.norm(dim=0, keepdim=True) + 1e-8, min=1.0)
        inner_products = inner_products * (1.0 / (inner_products.norm(dim=0, keepdim=True) + 1e-8)).clamp(max=1.0)
        # print(f"[debugging] see clipped normalized {inner_products=}")
        execution_logger.info(f"clipped normalized {inner_products=}")
        # inner_products = inner_products / (inner_products.norm(dim=0, keepdim=True) + 1e-8)
        # print(f"[debugging] see normalized {inner_products=}")
        # execution_logger.info(f"normalized {inner_products=}")
        #### always clip before dp, not normalization ####
        inner_products = torch.sum(inner_products, dim=-1) # should be of shape [#train_sample]
        # print(f"[debugging] see summed up {inner_products=}")
        execution_logger.info(f"summed up {inner_products=}")
        inner_products += torch.tensor(np.random.normal(scale=noise_multiplier*1.0*np.sqrt(2), size=inner_products.size(0))).to(inner_products.device)
        # print(f"[debugging] see summed up and DP-ed {inner_products=}")
        execution_logger.info(f"summed up and DP-ed {inner_products=}")
        inner_products = inner_products / num_val
        ################### [[[[dp on inner_products]]]] ends ###################

        ################### [[[[dp on residual]]]] starts ###################
        #### always clip before dp, not normalization ####
        # residual = residual / torch.clamp(residual.norm(dim=0, keepdim=True) + 1e-8, min=0.2)  # shape: [#param, #val_sample]
        residual = residual * (0.2 / (residual.norm(dim=0, keepdim=True) + 1e-8)).clamp(max=1.0)
        # print(f"[debugging] see clipped normalized {residual=}")
        execution_logger.info(f"clipped normalized {residual=}")
        # residual = residual / (residual.norm(dim=0, keepdim=True) + 1e-8)
        # print(f"[debugging] see normalized {residual=}")
        # execution_logger.info(f"normalized {residual=}")
        #### always clip before dp, not normalization ####
        residual = torch.sum(residual, dim=-1) # should be of shape [#train_sample]
        # print(f"[debugging] see summed up {residual=}")
        execution_logger.info(f"summed up {residual=}")
        residual += torch.tensor(np.random.normal(scale=noise_multiplier*0.2*np.sqrt(2), size=residual.size(0))).to(residual.device)
        # print(f"[debugging] see summed up and DP-ed {residual=}")
        execution_logger.info(f"summed up and DP-ed {residual=}")
        residual = residual / num_val
        ################### [[[[dp on residual]]]] ends ###################
        average_approximate_grad = G @ inner_products + residual  # [#param, #train_sample] x [#train_sample, 1] + [#param] --> [#param]
    else:
        ################### [[[[dp on inner_products]]]] starts ###################
        #### always clip before dp, not normalization ####
        # inner_products = inner_products / torch.clamp(inner_products.norm(dim=0, keepdim=True) + 1e-8, min=1.0)
        inner_products = inner_products * (1.0 / (inner_products.norm(dim=0, keepdim=True) + 1e-8)).clamp(max=1.0)
        # print(f"[debugging] see clipped normalized {inner_products=}")
        execution_logger.info(f"clipped normalized {inner_products=}")
        # inner_products = inner_products / (inner_products.norm(dim=0, keepdim=True) + 1e-8)
        # print(f"[debugging] see normalized {inner_products=}")
        # execution_logger.info(f"normalized {inner_products=}")
        #### always clip before dp, not normalization ####
        inner_products = torch.sum(inner_products, dim=-1) # should be of shape [#train_sample]
        # print(f"[debugging] see summed up {inner_products=}")
        execution_logger.info(f"summed up {inner_products=}")
        inner_products += torch.tensor(np.random.normal(scale=noise_multiplier*1.0, size=inner_products.size(0))).to(inner_products.device)
        # print(f"[debugging] see summed up and DP-ed {inner_products=}")
        execution_logger.info(f"summed up and DP-ed {inner_products=}")
        inner_products = inner_products / num_val
        ################### [[[[dp on inner_products]]]] ends ###################

        average_approximate_grad = G @ inner_products

    del G
    # print(f"[debugging] check {train_sample_grad=} after 'del G'")
    torch.cuda.empty_cache()
    gc.collect()

    return average_approximate_grad, inner_products, real_average_grad



def get_sample_grad(
    model,
    tokenizer,
    optimizer_state_dict,
    train_dataset,
    val_dataset,
    output_dir,
    val_sample_ratio=0.2,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=5,
    learning_rate=5e-5,
    noise_multiplier=None,
    add_instruction=False, 
    instruction=None,
    metric_inverse_epsilon=1E-6,
    clip_or_normalize='normalize',
    noise_on_vote=True,
    approximate_strategy='opt',
    use_eigen=False,
    clip_norm=1.0,
):
    if add_instruction == False:
        instruction = None
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        # per_device_train_batch_size=8,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.08,
        weight_decay=0.01,
        prediction_loss_only=True,
        report_to=[],
        remove_unused_columns=False,
    )
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 1: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    # select out val_sample_ratio of sampels from val_dataset
    # val_sample_size = int(val_sample_ratio * len(val_dataset))
    # val_indices = random.sample(range(len(val_dataset)), val_sample_size)
    val_indices = []
    while len(val_indices) == 0:
        random_values = np.random.rand(len(val_dataset))
        val_indices = [i for i, v in enumerate(random_values) if v <= val_sample_ratio]
    if isinstance(val_dataset, Dataset):
        val_dataset = val_dataset.select(val_indices)
    else:
        # If val_dataset is a dict or list of dicts
        if isinstance(val_dataset, dict):
            val_dataset = {k: [v[i] for i in val_indices] for k, v in val_dataset.items()}
        else:
            val_dataset = [val_dataset[i] for i in val_indices]
    
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    
    if 'ImageProcessor' in tokenizer.__class__.__name__:
        # Encode the training dataset if not already tokenized
        if not ("pixel_values" in train_dataset[0]):
            train_dataset = train_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        if not ("pixel_values" in val_dataset[0]):
            original_val_dataset = copy.deepcopy(val_dataset)
            val_dataset = val_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            val_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    else:
        # # Encode the training dataset if not already tokenized
        if not ("input_ids" in train_dataset[0]):
            if not add_instruction:
                train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt", return_overflowing_tokens=False)
                # Convert to HuggingFace Dataset if needed
                train_dataset = Dataset.from_dict(train_dataset)
            else:
                assert instruction is not None, "[ERROR] When specifying add_instruction=True, instruction should be provided."
                train_dataset = train_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=train_dataset.column_names
                                ) #, remove_columns=train_dataset["train"].column_names
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if not ("input_ids" in val_dataset[0]):
            if not add_instruction:
                original_val_dataset = copy.deepcopy(val_dataset)
                val_dataset = tokenizer(val_dataset['text'], truncation=True, padding=True, return_tensors="pt", return_overflowing_tokens=False)
                val_dataset = Dataset.from_dict(val_dataset)
            else:
                assert instruction is not None, "[ERROR] When specifying add_instruction=True, instruction should be provided."
                original_val_dataset = copy.deepcopy(val_dataset)
                val_dataset = val_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=val_dataset.column_names
                                ) #, remove_columns=val_dataset["train"].column_names
                val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    assert len(train_dataset) == train_dataset_size, f"[ERROR] overflowing length of val_dataset, should have tokenized {len(train_dataset)=} == {train_dataset_size=} before tokenization"
    assert len(val_dataset) == val_dataset_size, f"[ERROR] overflowing length of val_dataset, should have tokenized {len(val_dataset)=} == {val_dataset_size=} before tokenization"
    # print(f"[debugging] see dataset after tokenization: {len(train_dataset)=}, {len(val_dataset)=}")
    # print(f"[debugging] torch.cuda.memory, 2: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    if not 'ImageProcessor' in tokenizer.__class__.__name__:
        model.enable_input_require_grads()
        # print(f"[debugging] torch.cuda.memory, 2.5: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    model = GradSampleModule(model)
    # print(f"[debugging] torch.cuda.memory, 3: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    print("Sample-level Gradient calcultor prepared initialized successfully.")

    if 'ImageProcessor' in tokenizer.__class__.__name__:
        trainer = Trainer(
            model=model,
            args=training_args,
            # data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Use the same dataset for testing
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Use the same dataset for testing
        )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # param.requires_grad = True
    #         print(f"[debugging] Trainable parameter: {name}, shape: {param.shape}")
    # print(f"[debugging] torch.cuda.memory, 4: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # TODO: warmup_ratio=0.08, should be used with a scheduler, but since we are doing a single step update, I don't know if this should take effect
    if optimizer_state_dict != None:
        optimizer.load_state_dict(optimizer_state_dict)
    # print(f"[debugging] torch.cuda.memory, 5: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 6: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # Evaluate on this portion of private data for token prediction accuracy and loss
    eval_loss, eval_acc = evaluate_model_on_private_data(
        model=model,
        tokenizer=tokenizer,
        dataset=original_val_dataset,
        batch_size=8,
        add_instruction=add_instruction, instruction=instruction,
    )
    print(f"Evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")
    execution_logger.info(f"LLM evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")

    _accelerator = accelerate.Accelerator()
    trainer = _accelerator.prepare(trainer)
    model.train() # we don't accually train this, we just calculate the gradients
    # for name, param in model.named_parameters():
    #     print(f"checking  trainable parameter: {name}")
    #     if param.requires_grad:
    #         print(f"[debugging] Checking again, trainable parameter: {name}, shape: {param.shape}")
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 7: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # device = next(model.parameters()).device
    # Training loop
    for batch in trainer.get_train_dataloader():
        if type(batch) == type({}):
            labels = batch['labels']
        else:
            labels = batch.labels
        # for k in batch.keys():
        #     if k in _eval_total:
        #         batch[k] = torch.cat([batch[k], _eval_total[k]], dim=0)
        # print(f"[debugging] torch.cuda.memory, 9 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {batch.input_ids.shape=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        outputs = model(**batch) # have to change "hidden_state += residual" to "hidden_state = hidden_state + residual" in modeling_resnet.py of transformers to make it work for Images
        # loss = outputs.loss
        if 'ImageProcessor' in tokenizer.__class__.__name__:
            loss = per_sample_loss_function_image(labels, outputs.logits, reduction='none').sum()
        else:
            loss = per_sample_loss_function(labels, outputs.logits, reduction='none').sum()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"[debugging] torch.cuda.memory, 10 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()
    train_sample_grad = []
    # print(f"[debugging] torch.cuda.memory, 10-2 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    for name, param in model.named_parameters():
        if hasattr(param, "grad_sample"):
            grad_this_param = torch.concat(param.grad_sample)
            train_sample_grad.append(grad_this_param)
            del param.grad_sample
            # print(f"[debugging] checking {grad_this_param=} after 'del param.grad_sample'")
            param.grad_sample = []
    # print(f"[debugging] torch.cuda.memory, 10-3 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # model.zero_grad()
    train_sample_grad = [item.flatten(start_dim=1) for item in train_sample_grad]
    train_sample_grad = torch.cat(train_sample_grad, dim=1)

    torch.cuda.empty_cache()
    gc.collect()

    for batch in trainer.get_eval_dataloader():
        if type(batch) == type({}):
            labels = batch['labels']
        else:
            labels = batch.labels
        # print(f"[debugging] torch.cuda.memory, 9 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {batch.input_ids.shape=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        outputs = model(**batch)
        # loss = outputs.loss
        if 'ImageProcessor' in tokenizer.__class__.__name__:
            loss = per_sample_loss_function_image(labels, outputs.logits, reduction='none').sum()
        else:
            loss = per_sample_loss_function(labels, outputs.logits, reduction='none').sum()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"[debugging] torch.cuda.memory, 10 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()
    val_sample_grad = []
    # print(f"[debugging] torch.cuda.memory, 10-2 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    for name, param in model.named_parameters():
        if hasattr(param, "grad_sample"):
            grad_this_param = torch.concat(param.grad_sample)
            val_sample_grad.append(grad_this_param)
            del param.grad_sample
            # print(f"[debugging] checking {grad_this_param=} after 'del param.grad_sample'")
            param.grad_sample = []
    # print(f"[debugging] torch.cuda.memory, 10-3 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    val_sample_grad = [item.flatten(start_dim=1) for item in val_sample_grad]
    val_sample_grad = torch.cat(val_sample_grad, dim=1)
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {train_sample_grad.shape=}, {val_sample_grad.shape=}")
    
    torch.cuda.empty_cache()
    gc.collect()

    if approximate_strategy == 'opt':
        average_approximate_grad, G_coefficient, real_average_grad = opt_grad_combine(
            train_sample_grad, val_sample_grad, noise_multiplier=noise_multiplier, 
            metric_inverse_epsilon=metric_inverse_epsilon, 
            clip_or_normalize=clip_or_normalize, noise_on_vote=noise_on_vote,
            use_eigen=use_eigen,
            clip_norm=clip_norm,
        )
    elif 'GEP' in approximate_strategy:
        average_approximate_grad, G_coefficient, real_average_grad = gep(
            train_sample_grad, val_sample_grad, noise_multiplier=noise_multiplier, 
            with_residual=('wo' not in approximate_strategy), #, clip_or_normalize='clip'
            use_eigen=True,
            # clip_norm=clip_norm,
        )
    else:
        assert 'Residual' in approximate_strategy, f"[ERROR] {approximate_strategy=} not supported."
        average_approximate_grad, G_coefficient, real_average_grad = grad_projection(
            train_sample_grad, val_sample_grad, noise_multiplier=noise_multiplier, 
            with_residual=('wo' not in approximate_strategy), #, clip_or_normalize='clip'
            use_eigen=use_eigen,
            # clip_norm=clip_norm,
        )
    
    real_approximate_grad_cos_sim = torch.dot(real_average_grad, average_approximate_grad) / (torch.norm(real_average_grad) * torch.norm(average_approximate_grad))
    real_approximate_grad_l2_distance = torch.norm(real_average_grad - average_approximate_grad)
    # print(f"[debugging] {real_approximate_grad_cos_sim=}, avg_real_grad_norm={torch.norm(real_average_grad)}, {real_approximate_grad_l2_distance=} avg_approximate_grad_norm={torch.norm(average_approximate_grad)}, approximate/real norm ration={torch.norm(average_approximate_grad)/torch.norm(real_average_grad)}")
    execution_logger.info(f"{real_approximate_grad_cos_sim=}, avg_real_grad_norm={torch.norm(real_average_grad)}, {real_approximate_grad_l2_distance=}, avg_approximate_grad_norm={torch.norm(average_approximate_grad)}, approximate/real norm ration={torch.norm(average_approximate_grad)/torch.norm(real_average_grad)}")

    # apply the gradients back to the model.param to make sure that I can use optimizer.step() to update the model parameters
    param_numels = [param.numel() for name, param in model.named_parameters() if (hasattr(param, "grad_sample") and param.requires_grad)]
    grads_split = average_approximate_grad.split(param_numels)
    grad_idx = 0
    for name, param in model.named_parameters():
        if hasattr(param, "grad_sample") and param.requires_grad:
            param.grad = grads_split[grad_idx].view_as(param).clone()
            grad_idx += 1
            # print(f"[debugging] Will update parameter {name}: {param.data}")
    optimizer.step()
    optimizer.zero_grad()
    # for name, param in model.named_parameters():
    #     if hasattr(param, "grad_sample") and param.requires_grad:
    #         print(f"[debugging] See if parameter {name} is updated: {param.data}")


    # Evaluate on this portion of private data for token prediction accuracy and loss
    eval_loss, eval_acc = evaluate_model_on_private_data(
        model=model._module,
        tokenizer=tokenizer,
        dataset=original_val_dataset,
        batch_size=8,
        add_instruction=add_instruction, instruction=instruction,
    )
    print(f"After OptGrad fine-tune, evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")
    execution_logger.info(f"LLM after OptGrad fine-tune, evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")


    model._module.save_pretrained(output_dir) # Save the lora part
    tokenizer.save_pretrained(output_dir)

    # # TODO: bug, evaluation of multiple GPUs does not work properly
    # # eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")
    # eval_result = None

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    # del GTG
    # del GTG_inv
    # del G
    # del inner_products
    # # print(f"[debugging] check {train_sample_grad=} after 'del G'")
    torch.cuda.empty_cache()
    gc.collect()

    # print(f"[debugging] in <./pe/llm/sample_grad.py> 11 before axit: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # return model._module, G_coefficient, train_sample_grad, val_sample_grad, optimizer.state_dict()
    model = model.to_standard_module() # unwrap the model from GradSampleModule
    return model, G_coefficient, train_sample_grad, val_sample_grad, optimizer.state_dict()



def get_sample_grad_different_noise(
    model,
    tokenizer,
    optimizer_state_dict,
    train_dataset,
    val_dataset,
    output_dir,
    val_sample_ratio=0.2,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=5,
    learning_rate=5e-5,
    noise_multiplier=None,
    add_instruction=False, 
    instruction=None,
    metric_inverse_epsilon=1E-6,
    use_eigen=False,
    clip_norm=1.0,
):
    if add_instruction == False:
        instruction = None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        # per_device_train_batch_size=8,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.08,
        weight_decay=0.01,
        prediction_loss_only=True,
        report_to=[],
        remove_unused_columns=False,
    )
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 1: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    # select out val_sample_ratio of sampels from val_dataset
    # val_sample_size = int(val_sample_ratio * len(val_dataset))
    # val_indices = random.sample(range(len(val_dataset)), val_sample_size)
    val_indices = []
    while len(val_indices) == 0:
        random_values = np.random.rand(len(val_dataset))
        val_indices = [i for i, v in enumerate(random_values) if v <= val_sample_ratio]
    if isinstance(val_dataset, Dataset):
        val_dataset = val_dataset.select(val_indices)
    else:
        # If val_dataset is a dict or list of dicts
        if isinstance(val_dataset, dict):
            val_dataset = {k: [v[i] for i in val_indices] for k, v in val_dataset.items()}
        else:
            val_dataset = [val_dataset[i] for i in val_indices]
    
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    if 'ImageProcessor' in tokenizer.__class__.__name__:
        # Encode the training dataset if not already tokenized
        if not ("pixel_values" in train_dataset[0]):
            train_dataset = train_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        if not ("pixel_values" in val_dataset[0]):
            val_dataset = val_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            val_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    else:
        # # Encode the training dataset if not already tokenized
        if not ("input_ids" in train_dataset[0]):
            if not add_instruction:
                train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt", return_overflowing_tokens=False)
                # Convert to HuggingFace Dataset if needed
                train_dataset = Dataset.from_dict(train_dataset)
            else:
                assert instruction is not None, "[ERROR] When specifying add_instruction=True, instruction should be provided."
                train_dataset = train_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=train_dataset.column_names
                                ) #, remove_columns=train_dataset["train"].column_names
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if not ("input_ids" in val_dataset[0]):
            if not add_instruction:
                original_val_dataset = copy.deepcopy(val_dataset)
                val_dataset = tokenizer(val_dataset['text'], truncation=True, padding=True, return_tensors="pt", return_overflowing_tokens=False)
                val_dataset = Dataset.from_dict(val_dataset)
            else:
                assert instruction is not None, "[ERROR] When specifying add_instruction=True, instruction should be provided."
                original_val_dataset = copy.deepcopy(val_dataset)
                val_dataset = val_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=val_dataset.column_names
                                ) #, remove_columns=val_dataset["train"].column_names
                val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    assert len(train_dataset) == train_dataset_size, f"[ERROR] overflowing length of val_dataset, should have tokenized {len(train_dataset)=} == {train_dataset_size=} before tokenization"
    assert len(val_dataset) == val_dataset_size, f"[ERROR] overflowing length of val_dataset, should have tokenized {len(val_dataset)=} == {val_dataset_size=} before tokenization"
    # print(f"[debugging] see dataset after tokenization: {len(train_dataset)=}, {len(val_dataset)=}")
    # print(f"[debugging] torch.cuda.memory, 2: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")


    if not 'ImageProcessor' in tokenizer.__class__.__name__:
        model.enable_input_require_grads()
        # print(f"[debugging] torch.cuda.memory, 2.5: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    model = GradSampleModule(model)
    # print(f"[debugging] torch.cuda.memory, 3: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    print("Sample-level Gradient calcultor prepared initialized successfully.")

    if 'ImageProcessor' in tokenizer.__class__.__name__:
        trainer = Trainer(
            model=model,
            args=training_args,
            # data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Use the same dataset for testing
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Use the same dataset for testing
        )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # param.requires_grad = True
    #         print(f"[debugging] Trainable parameter: {name}, shape: {param.shape}")
    # print(f"[debugging] torch.cuda.memory, 4: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # TODO: warmup_ratio=0.08, should be used with a scheduler, but since we are doing a single step update, I don't know if this should take effect
    if optimizer_state_dict != None:
        optimizer.load_state_dict(optimizer_state_dict)
    # print(f"[debugging] torch.cuda.memory, 5: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 6: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # Evaluate on this portion of private data for token prediction accuracy and loss
    eval_loss, eval_acc = evaluate_model_on_private_data(
        model=model,
        tokenizer=tokenizer,
        dataset=original_val_dataset,
        batch_size=8,
        add_instruction=add_instruction, instruction=instruction,
    )
    print(f"Evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")
    execution_logger.info(f"LLM evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")

    _accelerator = accelerate.Accelerator()
    trainer = _accelerator.prepare(trainer)
    model.train() # we don't accually train this, we just calculate the gradients
    # for name, param in model.named_parameters():
    #     print(f"checking  trainable parameter: {name}")
    #     if param.requires_grad:
    #         print(f"[debugging] Checking again, trainable parameter: {name}, shape: {param.shape}")
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 7: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # device = next(model.parameters()).device
    # Training loop
    for batch in trainer.get_train_dataloader():
        if type(batch) == type({}):
            labels = batch['labels']
        else:
            labels = batch.labels
        # for k in batch.keys():
        #     if k in _eval_total:
        #         batch[k] = torch.cat([batch[k], _eval_total[k]], dim=0)
        # print(f"[debugging] torch.cuda.memory, 9 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {batch.input_ids.shape=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        outputs = model(**batch)
        # loss = outputs.loss
        if 'ImageProcessor' in tokenizer.__class__.__name__:
            loss = per_sample_loss_function_image(labels, outputs.logits, reduction='none').sum()
        else:
            loss = per_sample_loss_function(labels, outputs.logits, reduction='none').sum()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"[debugging] torch.cuda.memory, 10 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()
    train_sample_grad = []
    # print(f"[debugging] torch.cuda.memory, 10-2 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    for name, param in model.named_parameters():
        if hasattr(param, "grad_sample"):
            grad_this_param = torch.concat(param.grad_sample)
            train_sample_grad.append(grad_this_param)
            del param.grad_sample
            # print(f"[debugging] checking {grad_this_param=} after 'del param.grad_sample'")
            param.grad_sample = []
    # print(f"[debugging] torch.cuda.memory, 10-3 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # model.zero_grad()
    train_sample_grad = [item.flatten(start_dim=1) for item in train_sample_grad]
    train_sample_grad = torch.cat(train_sample_grad, dim=1)

    torch.cuda.empty_cache()
    gc.collect()

    for batch in trainer.get_eval_dataloader():
        if type(batch) == type({}):
            labels = batch['labels']
        else:
            labels = batch.labels
        # print(f"[debugging] torch.cuda.memory, 9 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {batch.input_ids.shape=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        outputs = model(**batch)
        # loss = outputs.loss
        if 'ImageProcessor' in tokenizer.__class__.__name__:
            loss = per_sample_loss_function_image(labels, outputs.logits, reduction='none').sum()
        else:
            loss = per_sample_loss_function(labels, outputs.logits, reduction='none').sum()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"[debugging] torch.cuda.memory, 10 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()
    val_sample_grad = []
    # print(f"[debugging] torch.cuda.memory, 10-2 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    for name, param in model.named_parameters():
        if hasattr(param, "grad_sample"):
            grad_this_param = torch.concat(param.grad_sample)
            val_sample_grad.append(grad_this_param)
            del param.grad_sample
            # print(f"[debugging] checking {grad_this_param=} after 'del param.grad_sample'")
            param.grad_sample = []
    # print(f"[debugging] torch.cuda.memory, 10-3 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    val_sample_grad = [item.flatten(start_dim=1) for item in val_sample_grad]
    val_sample_grad = torch.cat(val_sample_grad, dim=1)
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {train_sample_grad.shape=}, {val_sample_grad.shape=}")
    
    torch.cuda.empty_cache()
    gc.collect()

    # Let G = train_sample_grad.T
    G = train_sample_grad.T  # shape: [#param, #train_sample]
    eps = metric_inverse_epsilon  # small epsilon for numerical stability
    GTG = G.T @ G  # shape: [#train_sample, #train_sample]
    GTG = GTG + eps * torch.eye(GTG.shape[0], device=GTG.device, dtype=GTG.dtype)  # add epsilon to diagonal
    GTG_inv = torch.linalg.pinv(GTG)  # shape: [#train_sample, #train_sample]
    # # G_GTG_inv = G @ GTG_inv @ G.T # shape: [#param, #train_sample] * [#train_sample, #train_sample] * [#train_sample, #param] = [#param, #param]
    # G_GTG_inv = G @ GTG_inv # shape: [#param, #train_sample] * [#train_sample, #train_sample] = [#param, #train_sample]

    # Inner product of each row of train_sample_grad and each row of val_sample_grad
    # Result shape: [#train_sample, #val_sample]
    inner_products = train_sample_grad @ val_sample_grad.T

    # execution_logger.info(f"[debugging] {GTG.shape=}, {GTG=}")
    # execution_logger.info(f"[debugging] {GTG_inv.shape=}, {GTG_inv=}")
    # execution_logger.info(f"[debugging] {inner_products.shape=}, {inner_products=}")

    # ################### [[[[dp on G_GTG_inv @ inner_products]]]] starts ###################
    # # # final_grad: G_GTG_inv @ inner_products
    # # # Result shape: [#param, #val_sample]
    # # approximate_grad = G_GTG_inv @ inner_products

    # # G_coefficient for multiplying G
    # # Result shape: [#train_sample, #val_sample]
    # G_coefficient = GTG_inv @ inner_products

    # # add noise on to G_coefficient
    # print(f"[debugging] see non-normalized {G_coefficient=}")
    # execution_logger.info(f"non-normalized {G_coefficient=}")
    # G_coefficient = G_coefficient / (G_coefficient.norm(dim=0, keepdim=True) + 1e-8)
    # print(f"[debugging] see normalized {G_coefficient=}")
    # execution_logger.info(f"normalized {G_coefficient=}")
    # G_coefficient = torch.sum(G_coefficient, dim=-1) # should be of shape [#train_sample]
    # print(f"[debugging] see summed up {G_coefficient=}")
    # execution_logger.info(f"summed up {G_coefficient=}")
    # G_coefficient += torch.tensor(np.random.normal(scale=noise_multiplier, size=G_coefficient.size(0))).to(G_coefficient.device)
    # print(f"[debugging] see summed up and DP-ed {G_coefficient=}")
    # execution_logger.info(f"summed up and DP-ed {G_coefficient=}")
    # ################### [[[[dp on G_GTG_inv @ inner_products]]]] ends ###################


    execution_logger.info(f"{inner_products.norm(dim=0, keepdim=True)=}")
    print(f"inner_products.norm mean={torch.mean(inner_products.norm(dim=0, keepdim=True))}, std={torch.std(inner_products.norm(dim=0, keepdim=True))}, min={torch.min(inner_products.norm(dim=0, keepdim=True))}, max={torch.max(inner_products.norm(dim=0, keepdim=True))}")
    execution_logger.info(f"inner_products.norm mean={torch.mean(inner_products.norm(dim=0, keepdim=True))}, std={torch.std(inner_products.norm(dim=0, keepdim=True))}, min={torch.min(inner_products.norm(dim=0, keepdim=True))}, max={torch.max(inner_products.norm(dim=0, keepdim=True))}")
    ################### [[[[dp on inner_products]]]] starts ###################
    inner_products = inner_products / (inner_products.norm(dim=0, keepdim=True) + 1e-8)
    # print(f"[debugging] see normalized {inner_products=}")
    execution_logger.info(f"normalized {inner_products=}")
    inner_products = torch.sum(inner_products, dim=-1) # should be of shape [#train_sample]
    # print(f"[debugging] see summed up {inner_products=}")
    execution_logger.info(f"summed up {inner_products=}")
    inner_products += torch.tensor(np.random.normal(scale=noise_multiplier, size=inner_products.size(0))).to(inner_products.device)
    # print(f"[debugging] see summed up and DP-ed {inner_products=}")
    execution_logger.info(f"summed up and DP-ed {inner_products=}")
    G_coefficient = GTG_inv @ inner_products
    # print(f"[debugging] with dp-ed inner_products, {G_coefficient=}")
    execution_logger.info(f"with dp-ed inner_products, {G_coefficient=}")
    ################### [[[[dp on inner_products]]]] ends ###################

    approximate_grad = G @ G_coefficient # shape: [#param]
    average_approximate_grad = approximate_grad / inner_products.size(-1) # shape: [#param]
    print(f"{average_approximate_grad.shape=}")

    # apply the gradients back to the model.param to make sure that I can use optimizer.step() to update the model parameters
    param_numels = [param.numel() for name, param in model.named_parameters() if (hasattr(param, "grad_sample") and param.requires_grad)]
    grads_split = average_approximate_grad.split(param_numels)
    grad_idx = 0
    for name, param in model.named_parameters():
        if hasattr(param, "grad_sample") and param.requires_grad:
            param.grad = grads_split[grad_idx].view_as(param).clone()
            grad_idx += 1
            # print(f"[debugging] Will update parameter {name}: {param.data}")
    optimizer.step()
    optimizer.zero_grad()
    # for name, param in model.named_parameters():
    #     if hasattr(param, "grad_sample") and param.requires_grad:
    #         print(f"[debugging] See if parameter {name} is updated: {param.data}")


    # Evaluate on this portion of private data for token prediction accuracy and loss
    eval_loss, eval_acc = evaluate_model_on_private_data(
        model=model._module,
        tokenizer=tokenizer,
        dataset=original_val_dataset,
        batch_size=8,
        add_instruction=add_instruction, instruction=instruction,
    )
    print(f"After OptGrad fine-tune, evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")
    execution_logger.info(f"LLM after OptGrad fine-tune, evaluation on seleted {val_sample_ratio*100}% private data - Loss: {eval_loss:.5f}, Token Accuracy: {eval_acc:.5f}")


    model._module.save_pretrained(output_dir) # Save the lora part
    tokenizer.save_pretrained(output_dir)

    # # TODO: bug, evaluation of multiple GPUs does not work properly
    # # eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")
    # eval_result = None

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    del GTG
    del GTG_inv
    del G
    del inner_products
    # print(f"[debugging] check {train_sample_grad=} after 'del G'")
    torch.cuda.empty_cache()
    gc.collect()

    # print(f"[debugging] in <./pe/llm/sample_grad.py> 11 before axit: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # return model._module, G_coefficient, train_sample_grad, val_sample_grad, optimizer.state_dict()
    model = model.to_standard_module() # unwrap the model from GradSampleModule
    return model, G_coefficient, train_sample_grad, val_sample_grad, optimizer.state_dict()


def get_per_sample_loss(labels, logits):
    print(f"in <DPSDA-base/pe/llm/sample_grad.py>, {logits=}, {labels=}, {logits.shape=}, {labels.shape=}")
    # # For causual-LLM, shift so that tokens < n predict n
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {labels.shape=}, {logits.shape=}")
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {labels=}")
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {logits=}")
    # shift_labels = labels[..., 1:].contiguous()
    # shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = shift_labels[..., 1:].contiguous()
    shift_logits = logits
    print(f"in <DPSDA-base/pe/llm/sample_grad.py>, {shift_logits=}, {shift_labels=}, {shift_logits.shape=}, {shift_labels.shape=}")
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-100)
    # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)).to(dtype=torch.float32),  # [batch*seq, vocab]
        shift_labels.view(-1)                          # [batch*seq]
    )
    loss_per_token = loss_per_token.view(shift_labels.size())  # [batch, seq_len]
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {loss_per_token.shape=}")
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {loss_per_token=}")
    # Mask ignored tokens (e.g., padding with -100)
    mask = shift_labels != -100
    loss_per_token = loss_per_token * mask
    # Average over tokens per sample → per-sample loss
    loss_per_sample = loss_per_token.sum(dim=1) / mask.sum(dim=1)
    loss_per_sample = loss_per_sample.view(-1)  # Flatten to [batch_size]
    print(f"{loss_per_sample=}, {loss_per_token.sum()/mask.sum()}")
    return loss_per_sample


'''
# get_sample_grad_ghostsuite_version
def get_sample_grad(
    model,
    tokenizer,
    optimizer_state_dict,
    train_dataset,
    val_dataset,
    output_dir,
    val_sample_ratio=0.2,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=5,
    learning_rate=5e-5,
    noise_multiplier=None,
    add_instruction=False, 
    instruction=None,
    metric_inverse_epsilon=1E-6,
):
    if add_instruction == False:
        instruction = None
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     overwrite_output_dir=True,
    #     num_train_epochs=num_train_epochs,
    #     per_device_train_batch_size=per_device_train_batch_size,
    #     per_device_eval_batch_size=per_device_train_batch_size,
    #     save_steps=save_steps,
    #     logging_steps=logging_steps,
    #     learning_rate=learning_rate,
    #     warmup_ratio=0.08,
    #     weight_decay=0.01,
    #     prediction_loss_only=True,
    #     report_to=[],
    #     remove_unused_columns=False,
    # )
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 1: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    # select out val_sample_ratio of sampels from val_dataset
    # val_sample_size = int(val_sample_ratio * len(val_dataset))
    # val_indices = random.sample(range(len(val_dataset)), val_sample_size)
    val_indices = []
    while len(val_indices) == 0:
        random_values = np.random.rand(len(val_dataset))
        val_indices = [i for i, v in enumerate(random_values) if v <= val_sample_ratio]
    if isinstance(val_dataset, Dataset):
        val_dataset = val_dataset.select(val_indices)
    else:
        # If val_dataset is a dict or list of dicts
        if isinstance(val_dataset, dict):
            val_dataset = {k: [v[i] for i in val_indices] for k, v in val_dataset.items()}
        else:
            val_dataset = [val_dataset[i] for i in val_indices]
    
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    # # Encode the training dataset if not already tokenized
    if not ("labels" in train_dataset[0]):
        if not add_instruction:
            train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt", return_overflowing_tokens=False)
            # Convert to HuggingFace Dataset if needed
            train_dataset = Dataset.from_dict(train_dataset)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
        else:
            assert instruction is not None, "[ERROR] When specifying add_instruction=True, instruction should be provided."
            train_dataset = train_dataset.map(
                                chat_template_tokenize_example, 
                                fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                remove_columns=train_dataset.column_names
                            ) #, remove_columns=train_dataset["train"].column_names
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
            # data_collator = DefaultDataCollator(
            #     return_tensors="pt"
            # )
    if not ("labels" in val_dataset[0]):
        if not add_instruction:
            val_dataset = tokenizer(val_dataset['text'], truncation=True, padding=True, return_tensors="pt", return_overflowing_tokens=False)
            val_dataset = Dataset.from_dict(val_dataset)
        else:
            assert instruction is not None, "[ERROR] When specifying add_instruction=True, instruction should be provided."
            original_val_dataset = copy.deepcopy(val_dataset)
            val_dataset = val_dataset.map(
                                chat_template_tokenize_example, 
                                fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                remove_columns=val_dataset.column_names
                            ) #, remove_columns=val_dataset["train"].column_names
    assert len(train_dataset) == train_dataset_size, f"[ERROR] overflowing length of val_dataset, should have tokenized {len(train_dataset)=} == {train_dataset_size=} before tokenization"
    assert len(val_dataset) == val_dataset_size, f"[ERROR] overflowing length of val_dataset, should have tokenized {len(val_dataset)=} == {val_dataset_size=} before tokenization"
    # print(f"[debugging] see dataset after tokenization: {len(train_dataset)=}, {len(val_dataset)=}")
    # print(f"[debugging] {train_dataset.column_names=}")
    # for i in range(5):
    #     print(f"[debugging] {train_dataset['input_ids'][i]=}")
    #     print(f"[debugging] {train_dataset['attention_mask'][i]=}")
    #     print(f"[debugging] {train_dataset['labels'][i]=}")
    # for item in train_dataset['labels']:
    #     if int(item[0]) != -100:
    #         print(f"[debugging] in <./pe/llm/sample_grad.py>, found a label that is not -100: {item}")
    # print(f"[debugging] torch.cuda.memory, 2: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    model.enable_input_require_grads()
    # print(f"[debugging] torch.cuda.memory, 3: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,  # Use the same dataset for testing
    # )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # param.requires_grad = True
    #         print(f"[debugging] Trainable parameter: {name}, shape: {param.shape}")
    # print(f"[debugging] torch.cuda.memory, 4: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # TODO: warmup_ratio=0.08, should be used with a scheduler, but since we are doing a single step update, I don't know if this should take effect
    if optimizer_state_dict != None:
        optimizer.load_state_dict(optimizer_state_dict)
    # print(f"[debugging] torch.cuda.memory, 5: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    engine = AccumulateGradDotProdEngine(
        module=model,
        train_data_size=len(train_dataset),
        val_data_size=len(val_dataset),
        loss_reduction='mean', # or 'sum'
        average_grad=True, # if average_grad=True, then mean is used
        origin_params=None,
    )
    engine.attach(optimizer) # add forward and backward hooks

    torch.cuda.empty_cache()
    gc.collect()

    # print(f"[debugging] torch.cuda.memory, 6: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    _accelerator = accelerate.Accelerator()
    # trainer = _accelerator.prepare(trainer)
    model, train_dataset, val_dataset = _accelerator.prepare(model, train_dataset, val_dataset)
    model.train() # we don't accually train this, we just calculate the gradients
    # for name, param in model.named_parameters():
    #     print(f"checking  trainable parameter: {name}")
    #     if param.requires_grad:
    #         print(f"[debugging] Checking again, trainable parameter: {name}, shape: {param.shape}")
    # print(f"[debugging] torch.cuda.memory, 7: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # Training loop
    per_train_sample_loss = []
    device = next(model.parameters()).device
    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    # print(f"[debugging] in <./pe/llm/sample_grad.py>, {device=}")
    # for batch in trainer.get_train_dataloader():
    for batch in train_loader:
        if type(batch) == type({}):
            labels = batch['labels'].to(device)
        else:
            labels = batch.labels.to(device)
        print(f"{batch=}")
        # for k in batch.keys():
        #     print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {k=}, {batch[k].shape=}, {batch[k]=}")
        # print(f"[debugging] torch.cuda.memory, 9 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {labels.shape=}")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {labels=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

        outputs = model(**batch)
        # _built_in_loss = outputs.loss
        _per_sample_loss = per_sample_loss_function(labels, outputs.logits, shift_label=True, reduction='none')
        per_train_sample_loss.append(_per_sample_loss)
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {_built_in_loss=}, {torch.mean(_per_sample_loss)=}, {_per_sample_loss=}")
        optimizer.zero_grad()
        _per_sample_loss.mean().backward()
        # engine.prepare_gradients()
        # # optimizer.step()
        # engine.aggregate_and_log()
        # engine.clear_gradients()
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"[debugging] torch.cuda.memory, 10 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    per_train_sample_loss = torch.concat(per_train_sample_loss) # shape: [#train_sample]
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] torch.cuda.memory, 10-2 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    train_sample_layer_activation = []
    train_sample_layer_grad_output = []
    # for name, param in model.named_parameters():
    for name, layer in model.named_modules():
        # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, {name=}, {hasattr(layer, 'activations')=}, {hasattr(layer, 'grad_output')=}")
        if hasattr(layer, "activations") and hasattr(layer, "grad_output"):
            # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, {name=}, {len(layer.activations)=}, {layer.activations[0].shape=}, {type(layer.activations[0])=}, {layer.activations[0].device=}")
            # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, {name=}, ,{len(layer.grad_output)=}, {layer.grad_output[0].shape=}, {type(layer.grad_output[0])=}, {layer.grad_output[0].device=}")
            # # Check number of parameters in this layer
            # num_params = sum(p.numel() for p in layer.parameters(recurse=False))
            # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, number of parameters in layer '{name}': {num_params}")

            # activation_this_layer = torch.concat(layer.grad_output)
            train_sample_layer_activation.append(torch.concat(layer.activations,dim=0)) # shape: [#train_sample, #activation_dim]
            del layer.activations
            layer.activations = []
            # grad_this_layer = torch.concat(layer.grad_output)
            train_sample_layer_grad_output.append(torch.concat(layer.grad_output,dim=0)) # shape: [#train_sample, #grad_dim]
            del layer.grad_output
            layer.grad_output = []

    offload_hook.offload()
    gc.collect()
    torch.cuda.empty_cache()
    # print(f"[debugging] torch.cuda.memory, 10-3 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # TODO: calculate sample dot product
    GTG = engine.prepare_dotprod(
        model=model,
        activations=[train_sample_layer_activation], #, train_sample_layer_grad_output],  # shape: [#train_sample*2, #activation_dim]
        grad_outputs=[train_sample_layer_grad_output], #, train_sample_layer_grad_output], # shape: [#train_sample*2, #grad_dim]
        val_size=train_sample_layer_activation[0].size(0),  # equals to #train_sample
        device=device,
    ).to(dtype=torch.float)
    # print(f"[debugging] torch.cuda.memory, 10-4 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # print(f"[debugging] in <./pe/llm/sample_grad.py> {GTG=} of shape {GTG.shape}")
    # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, prepared sample dot product for layer '{name}' with {layer.weight.grad_dot_prod=} of shape {layer.weight.grad_dot_prod.shape}")
    # TODO: GTG should be of shape [#train_sample*2, #train_sample*2]
    model.zero_grad()
    # train_sample_grad = [item.flatten(start_dim=1) for item in train_sample_grad]
    # train_sample_grad = torch.cat(train_sample_grad, dim=1) # shape: [#train_sample, #param]

    torch.cuda.empty_cache()
    gc.collect()

    # for batch in trainer.get_eval_dataloader():
    for batch in val_loader:
        if type(batch) == type({}):
            labels = batch['labels'].to(device)
        else:
            labels = batch.labels.to(device)
        # print(f"[debugging] torch.cuda.memory, 9 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch, {batch.input_ids.shape=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        outputs = model(**batch)
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, {next(model.parameters()).device=}")
        # loss = outputs.loss
        if 'ImageProcessor' in tokenizer.__class__.__name__:
            loss = per_sample_loss_function_image(labels, outputs.logits, reduction='none').sum()
        else:
            loss = per_sample_loss_function(labels, outputs.logits, reduction='none').sum()
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"[debugging] torch.cuda.memory, 10 (in val iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    offload_hook.offload()
    torch.cuda.empty_cache()
    gc.collect()
    val_sample_layer_activation = []
    val_sample_layer_grad_output = []
    # for name, param in model.named_parameters():
    for name, layer in model.named_modules():
        # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, {name=}, {hasattr(layer, 'activations')=}, {hasattr(layer, 'grad_output')=}")
        if hasattr(layer, "activations") and hasattr(layer, "grad_output"): # TODO HERE
            # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, {name=}, {len(layer.activations)=}, {layer.activations[0].shape=}, {type(layer.activations[0])=}, {layer.activations[0].device=}")
            # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, {name=}, ,{len(layer.grad_output)=}, {layer.grad_output[0].shape=}, {type(layer.grad_output[0])=}, {layer.grad_output[0].device=}")
            # # Check number of parameters in this layer
            # num_params = sum(p.numel() for p in layer.parameters(recurse=False))
            # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, number of parameters in layer '{name}': {num_params}")

            # activation_this_layer = torch.concat(layer.grad_output)
            val_sample_layer_activation.append(torch.concat(layer.activations,dim=0)) # shape: [#val_sample, #activation_dim]
            del layer.activations
            layer.activations = []
            # grad_this_layer = torch.concat(layer.grad_output)
            val_sample_layer_grad_output.append(torch.concat(layer.grad_output,dim=0)) # shape: [#val_sample, #grad_dim]
            del layer.grad_output
            layer.grad_output = []
    
    torch.cuda.empty_cache()
    gc.collect()
    # TODO: calculate sample dot product
    inner_products = engine.prepare_dotprod(
        model=model,
        activations=[train_sample_layer_activation, val_sample_layer_activation], #, train_sample_layer_grad_output],  # shape: [#train_sample*2, #activation_dim]
        grad_outputs=[train_sample_layer_grad_output, val_sample_layer_grad_output], #, train_sample_layer_grad_output], # shape: [#train_sample*2, #grad_dim]
        val_size=val_sample_layer_activation[0].size(0),  # equals to #train_sample
        device=device,
    ).to(dtype=torch.float)
    # print(f"[debugging] in <./pe/llm/sample_grad.py> ghost version train, prepared sample dot product for layer '{name}' with {layer.weight.grad_dot_prod=} of shape {layer.weight.grad_dot_prod.shape}")
    # TODO: inner_products should be of shape [#train_sample+#val_sample, #train_sample+#val_sample]
    # print(f"[debugging] in <./pe/llm/sample_grad.py> {inner_products=} of shape {inner_products.shape}")

    engine.detach() # remove forward and backward hooks
    gc.collect()
    torch.cuda.empty_cache()

    # ########### for ghost suit sample doc version ###########
    eps = metric_inverse_epsilon  # small epsilon for numerical stability
    GTG = GTG + eps * torch.eye(GTG.shape[0], device=GTG.device, dtype=GTG.dtype)  # add epsilon to diagonal
    GTG_inv = torch.linalg.pinv(GTG)  # shape: [#train_sample, #train_sample]
    
    # G_coefficient for multiplying loss_vector
    # Result shape: [#train_sample, #val_sample]
    G_coefficient = GTG_inv @ inner_products
    # ########### for ghost suit sample doc version ###########

    # add noise on to G_coefficient
    # print(f"[debugging] see non-normalized {G_coefficient=}")
    # execution_logger.info(f"[debugging] see non-normalized {G_coefficient=}")
    G_coefficient = G_coefficient / (G_coefficient.norm(dim=0, keepdim=True) + 1e-8)
    # print(f"[debugging] see normalized {G_coefficient=}")
    # execution_logger.info(f"[debugging] see normalized {G_coefficient=}")
    G_coefficient = torch.sum(G_coefficient, dim=-1) # should be of shape [#train_sample]
    # print(f"[debugging] see summed up {G_coefficient=}")
    # execution_logger.info(f"[debugging] see summed up {G_coefficient=}")
    G_coefficient += torch.tensor(np.random.normal(scale=noise_multiplier, size=G_coefficient.size(0))).to(G_coefficient.device)
    # print(f"[debugging] see summed up and DP-ed {G_coefficient=}")
    # execution_logger.info(f"[debugging] see summed up and DP-ed {G_coefficient=}")

    # ########### for ghost suit sample doc version ###########
    # ### get loss again for each training sample ###
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     collate_fn=data_collator
    # )
    for name, param in model.named_parameters():
        if param.initially_requires_grad:
            param.requires_grad = True
            # print(f"[debugging] Initial requires grad {name}: {param.data=}")
    # per_train_sample_loss = []
    # for batch in trainer.get_train_dataloader():
    # with torch.no_grad():
    start = 0
    for batch in train_loader:
        if type(batch) == type({}):
            labels = batch['labels'].to(device)
        else:
            labels = batch.labels.to(device)
        # print(f"[debugging] torch.cuda.memory, 9 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch=1, {labels.shape=}")
        # print(f"[debugging] in <./pe/llm/sample_grad.py>, for batch=1, {labels=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        outputs = model(**batch)
        # per_train_sample_loss.append(outputs.loss)
        _per_sample_loss = per_sample_loss_function(labels, outputs.logits, shift_label=True, reduction='none')
        loss = (_per_sample_loss * G_coefficient[start:start+len(_per_sample_loss)] / inner_products.size(-1)).sum() # shape: [1]
        loss.backward() # calculate the gradients for the model parameters
        start += len(_per_sample_loss)
        # torch.cuda.empty_cache()
        # gc.collect()
        # print(f"[debugging] torch.cuda.memory, 10 (in train iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    print(f"{per_train_sample_loss=}")
    # per_train_sample_loss = torch.stack(per_train_sample_loss).view(-1) # shape: [#train_sample]
    gc.collect()
    torch.cuda.empty_cache()

    # # weighted_loss = per_train_sample_loss * G_coefficient # shape: [#train_sample]
    # # approximate_loss = torch.sum(weighted_loss) / len(weighted_loss) # normalize the loss
    # # print(f"[debugging] in <./pe/llm/sample_grad.py> approximate_loss: {approximate_loss=}")
    # # approximate_loss.backward() # calculate the gradients for the model parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"[debugging] Will update parameter {name}: {param.data}")
    optimizer.step()
    optimizer.zero_grad()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"[debugging] See if parameter {name} is updated: {param.data}")
    # ########### for ghost suit sample doc version ###########


    model.save_pretrained(output_dir) # Save the lora part
    tokenizer.save_pretrained(output_dir)

    # # TODO: bug, evaluation of multiple GPUs does not work properly
    # # eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")
    # eval_result = None

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    torch.cuda.empty_cache()
    gc.collect()

    # print(f"[debugging] in <./pe/llm/sample_grad.py> 11 before axit: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    return model, G_coefficient, None, None # , train_sample_grad, val_sample_grad
    # return eval_result, trainer.model, tokenizer
'''


