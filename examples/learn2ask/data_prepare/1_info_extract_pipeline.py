import json
import time

from llm_info_extraction import LLM_info_extraction, parse_llm_output
from message_splitter import split_session_to_json_lines


def process_jsonl_file(
    input_file, output_file, model_call_mode="online_api", max_retries=3, **kwargs
):
    """
    Process all sessions in a JSONL file and save results based on specified output mode.

    Args:
        input_file (str): Path to input JSONL file
        output_mode (str): Either "single_file" or "multiple_files"
        output_file (str): Path to output file (required if output_mode="single_file")
        output_dir (str): Path to output directory (required if output_mode="multiple_files")
        model_call_mode (str): Either "online_api" or "local_vllm"
        max_retries (int): Maximum number of retries for LLM calls
        **kwargs: Additional parameters for API calls

    Returns:
        str: Success message or error information
    """
    try:
        # Read and process each session
        with open(input_file, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                if line.strip():
                    try:
                        session = json.loads(line)
                        print(
                            f"Processing session {session.get('session_id', 'unknown')} (line {line_num})..."
                        )

                        # Process the session
                        processed_lines = process_session(
                            session, model_call_mode, max_retries, **kwargs
                        )
                        for line in processed_lines:
                            with open(output_file, "a", encoding="utf-8") as outfile:
                                outfile.write(line + "\n")

                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    except Exception as e:
                        print(f"Warning: Error processing session at line {line_num}: {e}")

        return f"Successfully processed. Results saved to {output_file}"

    except Exception as e:
        return f"Error processing JSONL file: {str(e)}"


def process_session(session, model_call_mode="online_api", max_retries=3, **kwargs):
    """
    Pipeline function that splits messages into rounds and extracts info from each round's remaining chat.

    Args:
        session (dict): Session dictionary containing 'session_id', 'diagn', and 'messages' keys
        model_call_mode (str): Either "online_api" or "local_vllm"
        max_retries (int): Maximum number of retries for LLM calls
        **kwargs: Additional parameters for API calls

    Returns:
        list: List of JSON strings with added "info_set" key, or error information
    """
    try:
        # Step 1: Split messages into JSON lines
        json_lines = split_session_to_json_lines(session)

        # Step 2: Process each JSON line with LLM info extraction
        processed_lines = []

        for line in json_lines:
            data = json.loads(line)
            remaining_chat = data.get("remaining_chat", "")

            # Retry loop for LLM calls
            info_set = None
            for attempt in range(max_retries):
                try:
                    # Call LLM info extraction (using mock function for testing)
                    llm_response = LLM_info_extraction(remaining_chat, model_call_mode, **kwargs)

                    info_set = parse_llm_output(llm_response)

                    if isinstance(info_set, list):
                        break
                    else:
                        # If parsing failed, this is an error message
                        print(f"Attempt {attempt + 1} failed: {info_set}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with exception: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Shorter wait for testing

            data["info_set"] = info_set
            processed_lines.append(json.dumps(data, ensure_ascii=False))

        return processed_lines

    except Exception as e:
        return f"Pipeline error: {str(e)}"


# Example usage:
if __name__ == "__main__":
    input_file_path = "data_prepare_learn2ask/test_origin.jsonl"
    output_file_path = "data_prepare_learn2ask/test_processed.jsonl"
    process_jsonl_file(input_file=input_file_path, output_file=output_file_path)
