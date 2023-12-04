"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from tritonclient.utils import InferenceServerException
import tritonclient.grpc.aio as grpcclient

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    elif backend == "triton":
        params = {
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": "0.0" if use_beam_search else "1.0",
            "top_p": "1.0",
            "max_tokens": str(output_len),
            "ignore_eos": True,
        }
    elif backend == "tensort":
        params = {
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": "0.0" if use_beam_search else "1.0",
            "top_p": "1.0",
            "top_k": 1,
            "max_tokens": output_len,
            "ignore_eos": True,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if backend in ["vllm", "tgi"]:
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(api_url, headers=headers,
                                        json=pload) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                output = json.loads(output)

                # Re-send the request if it failed.
                if "error" not in output:
                    break
    elif backend == "triton":
        async with grpcclient.InferenceServerClient(
                url=api_url) as triton_client:

            async def async_request_iterator():
                try:
                    yield create_request_triton_vllm(prompt, False, 1, params,
                                                     "vllm_model")
                except Exception as error:
                    print(f"caught error in request iterator:  {error}")

            try:
                # Start streaming
                response_iterator = triton_client.stream_infer(
                    inputs_iterator=async_request_iterator(),
                    stream_timeout=None,
                )
                # Read response from the stream
                async for response in response_iterator:
                    result, error = response
                    if error:
                        raise error
                    for chunk in result.as_numpy("text_output"):
                        _ = chunk

            except InferenceServerException as error:
                print(f"caught error in request iterator:  {error}")

    elif backend == "tensort":
        async with grpcclient.InferenceServerClient(
                url=api_url) as triton_client:

            async def async_request_iterator():
                try:
                    yield create_request_triton_tensortllm(
                        prompt, params, "ensemble")
                except Exception as error:
                    print(f"caught error in request iterator:  {error}")

            try:
                # Start streaming
                response_iterator = triton_client.stream_infer(
                    inputs_iterator=async_request_iterator(),
                    stream_timeout=None,
                )
                # Read response from the stream
                async for response in response_iterator:
                    result, error = response
                    if error:
                        raise error
                    for chunk in result.as_numpy("text_output"):
                        _ = chunk

            except InferenceServerException as error:
                print(f"caught error in request iterator:  {error}")

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


def create_request_triton_vllm(prompt, stream, request_id, sampling_parameters,
                               model_name):
    inputs = []
    prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
    try:
        inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(prompt_data)
    except Exception as e:
        print(f"Encountered an error {e}")
    stream_data = np.array([stream], dtype=bool)
    inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(stream_data)

    # Add requested outputs
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("text_output"))

    # Issue the asynchronous sequence inference.
    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": str(request_id),
        "parameters": sampling_parameters,
    }


def create_request_triton_tensortllm(prompt, sampling_parameters, model_name):
    inputs = []
    prompt_data = np.array([[prompt.encode("utf-8")]], dtype=np.object_)
    # prompt_data = prompt_data.reshape(prompt_data.shape)
    # print(prompt_data.shape)
    empty_string_data = np.zeros((1, 1), dtype=np.object_)
    max_tokens_data = np.array([[sampling_parameters['max_tokens']]],
                               np.uint32)
    top_p_data = np.array([[sampling_parameters['top_p']]], np.float32)
    top_k_data = np.array([[sampling_parameters['top_k']]], np.uint32)
    temperature_data = np.array([[sampling_parameters['temperature']]],
                                np.float32)
    try:
        inputs.append(grpcclient.InferInput("text_input", [1, 1], "BYTES"))
        inputs[-1].set_data_from_numpy(prompt_data)

        inputs.append(
            grpcclient.InferInput("max_tokens", max_tokens_data.shape,
                                  "UINT32"))
        inputs[-1].set_data_from_numpy(max_tokens_data)

        inputs.append(
            grpcclient.InferInput("bad_words", empty_string_data.shape,
                                  "BYTES"))
        inputs[-1].set_data_from_numpy(empty_string_data)

        inputs.append(
            grpcclient.InferInput("stop_words", empty_string_data.shape,
                                  "BYTES"))
        inputs[-1].set_data_from_numpy(empty_string_data)

        inputs.append(grpcclient.InferInput("top_p", top_p_data.shape, "FP32"))
        inputs[-1].set_data_from_numpy(top_p_data)

        inputs.append(
            grpcclient.InferInput("top_k", top_p_data.shape, "UINT32"))
        inputs[-1].set_data_from_numpy(top_k_data)

        inputs.append(
            grpcclient.InferInput("temperature", top_p_data.shape, "FP32"))
        inputs[-1].set_data_from_numpy(temperature_data)
    except Exception as e:
        print(f"Encountered an error {e}")

    # Add requested outputs
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("text_output"))

    # Issue the asynchronous sequence inference.
    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
    }


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(backend, api_url, prompt, prompt_len, output_len,
                         best_of, use_beam_search))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"{args.host}:{args.port}" if args.backend in [
        "triton", "tensort"
    ] else f"http://{args.host}:{args.port}{args.host_path}"
    tokenizer = get_tokenizer(args.tokenizer,
                              trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(args.backend, api_url, input_requests, args.best_of,
                  args.use_beam_search, args.request_rate))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

    total_tokens = np.sum([(prompt_len + output_len)
                           for prompt_len, output_len, _ in REQUEST_LATENCY])

    total_output_tokens = np.sum(
        [output_len for _, output_len, _ in REQUEST_LATENCY])

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    print(f"Average tokens/s: {total_tokens/benchmark_time}")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")
    print(f"Average output tokens/s: {total_output_tokens/benchmark_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend",
                        type=str,
                        default="vllm",
                        choices=["vllm", "tgi", "triton", 'tensort'])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host_path", type=str, default="/generate")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer",
                        type=str,
                        required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of",
                        type=int,
                        default=1,
                        help="Generates `best_of` sequences per prompt and "
                        "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()
    main(args)
