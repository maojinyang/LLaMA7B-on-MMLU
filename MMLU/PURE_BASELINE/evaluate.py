# coding=utf-8

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Tuple
import os
import numpy as np
import pandas as pd
import time
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Tokenizer, Transformer, LLaMA
from crop import crop

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
        ckpt_dir: str,
        tokenizer_path: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def get_top_n_idx(nums, n):
    nums = copy.deepcopy(nums)
    result = []
    for i in range(n):
        idx = nums.index(max(nums))
        result.append(idx)
        nums[idx] = -1000000
    return result


def get_llama_vocabs_and_generator():
    ckpt_dir = "/users10/mjyang/RETRIEVAL_ENHANCE/LLAMA/MODEL/7B"
    tokenizer_path = "/users10/mjyang/RETRIEVAL_ENHANCE/LLAMA/RUN/tokenizer.model"
    max_seq_len = 3072
    max_batch_size = 32

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    tokenizer = Tokenizer(model_path=tokenizer_path)
    sp = tokenizer.sp_model
    assert sp.get_piece_size() == tokenizer.n_words
    vocabs = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    assert len(vocabs) == 32000

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    return generator, vocabs


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def eval(args, subject, engine, dev_df, test_df, generator, vocabs):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        item = {"prompt": prompt, "train_prompt": train_prompt, "prompt_end": prompt_end}
        results, probs = generator.generate([prompt], max_gen_len=1, temperature=0)
        token_prob = probs[0].cpu().tolist()[0]
        results_dict = {}
        index = get_top_n_idx(token_prob, 5)
        for idx in index:
            results_dict[vocabs[idx]] = token_prob[idx]

        answers = ["▁A", "▁B", "▁C", "▁D"]
        lprobs = []
        for answer in answers:
            try:
                lprobs.append(results_dict[answer])
            except:
                print("Warning: {} not found. Artificially adding log prob of -100.".format(answer))
                lprobs.append(-100)

        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    engines = args.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    print(subjects)
    print(args)
    generator, vocabs = get_llama_vocabs_and_generator()

    for engine in engines:
        print(engine)
        all_cors = []

        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, engine, dev_df, test_df, generator, vocabs)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--engine", "-e", choices=["llama"], default=["llama"], nargs="+")
    args = parser.parse_args()
    main(args)
