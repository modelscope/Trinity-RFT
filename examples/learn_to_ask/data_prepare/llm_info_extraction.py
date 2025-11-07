import os

import openai
import torch
import transformers

tokenizer = None
llm = None


def LLM_info_extraction(remaining_chat, model_call_mode, **kwargs):
    """
    Extract information from remaining_chat using LLM.

    Args:
        remaining_chat (str): The chat content to process
        model_call_mode (str): Either "online_api" or "local_vllm"
        **kwargs: Additional parameters for API calls

    Returns:
        str: Response text from LLM or error information
    """

    # Create messages format with system and user roles
    system_message = """
    # Task:
    You are a medical information assistant. Given a dialogue between a physician (assistant) and a patient (user), extract the clinical attributes of interest to the physician based on their questions. The target fields include: symptom, symptom nature, symptom location, symptom severity, and symptom trigger. Then, identify the corresponding specific information from the patient's responses and pair it with the respective field.
    # Requirements:
        - Do not fabricate information or introduce new fields not listed above. Ignore patient-reported information regarding prior medication use, allergies, or underlying comorbidities; do not include such details in the output.
        - Only include fields explicitly inquired about by the physician. Omit any fields not addressed in the dialogue. Avoid outputting vague terms (e.g., "unspecified" or "unknown").
        - Prevent duplication: if a symptom description already includes anatomical location, do not separately list the location field.
        - Format each entry as a string enclosed in single quotes ('), and separate multiple entries with commas, ensuring any necessary escape characters within the strings. Enclose the entire output within square brackets to form a list. If the dialogue is unrelated to the aforementioned clinical attributes, output only "[]".
        - Do not include reasoning steps or additional commentary outside the specified format. Condense colloquial patient expressions into concise, standardized, and clinically appropriate terminology.
    # Example output format:
    ['symptom: diarrhea', 'symptom nature: watery stool', 'symptom severity: 4-5 times per day']
    """
    user_message = remaining_chat

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "```\n" + user_message + "\n```\n"},
    ]

    try:
        if model_call_mode == "online_api":
            # OpenAI-style API call
            return _call_online_api(messages, **kwargs)
        elif model_call_mode == "local_vllm":
            # Local vLLM call
            return _call_local_vllm(messages, **kwargs)
        else:
            return f"Error: Invalid model_call_mode '{model_call_mode}'. Must be 'online_api' or 'local_vllm'."
    except Exception as e:
        return f"Error occurred: {str(e)}"


def _call_online_api(messages, **kwargs):
    """Handle OpenAI-style API calls"""
    # Extract API parameters from kwargs or use defaults
    api_key = kwargs.get("api_key", os.getenv("DASHSCOPE_API_KEY"))
    api_base = kwargs.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = kwargs.get("model", "qwen2.5-72b-instruct")
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 500)

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )

    return response.choices[0].message.content


def _call_local_vllm(messages, **kwargs):
    """Handle local vLLM calls"""
    try:
        from vllm import LLM, SamplingParams

        model_path = kwargs.get("model_path")
        if not model_path:
            return "Error: model_path is required for local vLLM inference"

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        top_p = kwargs.get("top_p", 0.9)
        repetition_penalty = kwargs.get("repetition_penalty", 1.1)

        # GPU/CUDA related parameters for vLLM
        tensor_parallel_size = kwargs.get("tensor_parallel_size", torch.cuda.device_count())
        gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        enforce_eager = kwargs.get("enforce_eager", False)
        dtype = kwargs.get("dtype", "auto")
        max_model_len = kwargs.get("max_model_len", 4096)

        # Initialize the LLM with the provided model path and GPU parameters
        global llm, tokenizer
        if llm is None:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                dtype=dtype,
                max_model_len=max_model_len,
            )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )

        # Convert messages to a single prompt string
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = llm.generate([prompt], sampling_params)

        return outputs[0].outputs[0].text

    except ImportError:
        return "Error: vLLM library not installed. Please install it with 'pip install vllm'"
    except Exception as e:
        return f"Error in local vLLM inference: {str(e)}"


def parse_llm_output(output_str):
    """
    Convert the LLM info extraction output string to a list of strings.

    Args:
        output_str (str): String in format "['symptom: diarrhea', 'symptom nature: watery stool', 'symptom severity: 4-5 times per day']"

    Returns:
        list: List of strings if successful, error message string if failed
    """
    import ast

    try:
        result = ast.literal_eval(output_str)
        if not isinstance(result, list):
            return f"Error: Expected a list, got {type(result)}"

        return result
    except Exception as e:
        return f"Error parsing output: [{repr(output_str)}] error = {str(e)}"
