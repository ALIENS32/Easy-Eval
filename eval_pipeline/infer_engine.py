import argparse
import concurrent
from concurrent.futures import ThreadPoolExecutor
import logging
import tqdm
from config.model_args_config import ModelArgsConfig
import os
import sys
from dataclasses import asdict
import threading

# Add the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from models.llm_factory import LLM
from utils import Utils
from config.data_template_config import DataTemplate


def get_answer(llm, query):
    """
    Retrieves a response from the LLM.
    Implements a retry mechanism for API calls.
    """
    messages = [
        {"role": "user", "content": query}
    ]
    for attempt in range(10):
        try:
            response = llm.get_response(messages)
            if "formal_answer" in response:
                logging.info(f"Successfully received response on attempt {attempt + 1}")
                return response["formal_answer"]
            else:
                logging.warning(f"Response missing 'formal_answer' key on attempt {attempt + 1}. Response: {response}")
                continue  # Continue to the next attempt if the key is missing
        except Exception as e:
            logging.warning(f"API Retry: Attempt {attempt + 1}/10 failed with error: {e}")
    logging.error(f"Failed to get response after 10 attempts for query: {query}")
    return ""

# Define a lock for protecting shared counters
shared_counts_lock = threading.Lock()

def generate_inference(llm, sample, shared_counts):
    """
    Generates an inference result for a single sample using the LLM.
    Updates shared_counts in a thread-safe manner.
    """
    pid = sample.get('pid')
    if pid is None:
        logging.error(f"Sample is missing 'pid' key. Skipping.")
        with shared_counts_lock:
            shared_counts["fail_sample_num"] += 1
        return {"pid": "N/A", "status": "ERROR", "inference_result": ""} # Return an error object

    try:
        # Increment processed sample count (thread-safe)
        with shared_counts_lock:
            shared_counts["processed_sample_num"] += 1
        
        logging.debug(f"Generating inference for sample PID: {pid}")
        
        # Construct the query for the LLM based on the sample data
        inference_query = DataTemplate.get_inference_query(sample)
        
        # Get the LLM's response
        llm_response = get_answer(llm, inference_query)

        # Create the inference result object
        inference_object = {
            "pid": pid,
            "status": "PASS" if llm_response else "FAIL", # Mark as FAIL if no response
            "inference_result": llm_response
        }

        if not llm_response:
            with shared_counts_lock:
                shared_counts["fail_sample_num"] += 1
            logging.warning(f"LLM did not return a response for PID: {pid}. Marked as FAIL.")

        return inference_object
    except Exception as e:
        logging.error(f"Error generating inference for sample PID {pid}: {e}")
        # Increment fail sample count (thread-safe)
        with shared_counts_lock:
            shared_counts["fail_sample_num"] += 1
        return {"pid": pid, "status": "ERROR", "inference_result": ""} # Mark as error


def infer_main(args):
    """
    Main function for orchestrating the inference generation process.
    Handles LLM initialization, data loading, parallel inference, and result saving.
    """
    logging.info("Starting inference generation process.")

    # Create ModelArgsConfig instance from parsed arguments
    model_args = asdict(ModelArgsConfig.from_argparse_args(args))
    model_args["model"] = args.model_name # Use model_name for LLM initialization
    try:
        llm = LLM(model_args)
        logging.info(f"LLM initialized with model arguments: {model_args}")
    except Exception as e:
        logging.critical(f"Failed to initialize LLM: {e}")
        sys.exit(1) # Exit if LLM cannot be initialized

    # Define paths for storing results
    save_path = os.path.join("results", args.model_name, "infer_result.jsonl")

    # Create directories if they don't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        logging.info(f"Created directory: {os.path.dirname(save_path)}")
    else:
        logging.info(f"Directory already exists: {os.path.dirname(save_path)}")

    # Load the dataset
    dataset_dict = Utils.load_data(args.dataset_name, args.split)
    all_dataset_samples = list(dataset_dict.values()) # Convert dict values to a list of samples

    # Load existing inference results for resumption
    all_existing_infer_results = []
    if os.path.exists(save_path):
        all_existing_infer_results = Utils.load_jsonl_data(save_path)
        logging.info(f"Loaded {len(all_existing_infer_results)} existing inference results from {save_path} for resuming.")

    # Filter dataset samples based on provided pid_list
    samples_to_process = []
    if args.pid_list:
        logging.info(f"Filtering dataset samples to process only PIDs: {args.pid_list}")
        for sample in all_dataset_samples:
            if sample.get("pid") in args.pid_list:
                samples_to_process.append(sample)
        if len(samples_to_process) != len(args.pid_list):
            logging.warning(f"Some PIDs from pid_list ({len(args.pid_list)}) were not found in the dataset ({len(samples_to_process)} found).")
    else:
        logging.info("No pid_list specified, processing all samples in the dataset.")
        samples_to_process = all_dataset_samples

    # Store all inference results (both historical and new) in a dictionary for easy updates and final saving
    final_infer_results_map = {res["pid"]: res for res in all_existing_infer_results if "pid" in res}

    # Filter out samples that have already been successfully processed
    new_samples_for_inference = []
    for sample in samples_to_process:
        pid = sample.get("pid")
        if pid in final_infer_results_map and final_infer_results_map[pid].get("status") == "PASS":
            logging.debug(f"PID {pid} already successfully inferred, skipping.")
        else:
            new_samples_for_inference.append(sample)

    logging.info(f"Will generate inference for {len(new_samples_for_inference)} new samples out of {len(samples_to_process)} total samples relevant to pid_list (considering existing successful inferences).")
    if all_existing_infer_results:
        logging.info("Incorporated existing results where available for resumed inference.")

    # Initialize shared inference counters
    shared_counts = {
        "processed_sample_num": 0,
        "fail_sample_num": 0,
    }
    logging.info("Shared counters initialized.")

    def worker(sample):
        """Worker function for the thread pool to generate inference for a single sample."""
        # Generate inference for the sample and update counts
        infer_result = generate_inference(llm, sample, shared_counts)
        return infer_result

    # Process samples using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.infer_proc) as executor:
        futures = [executor.submit(worker, sample) for sample in new_samples_for_inference]
        logging.info(f"Submitted {len(futures)} inference tasks to the thread pool with {args.infer_proc} workers.")
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating Inferences")):
            try:
                infer_result = future.result()
                # Update the final map with the result of the current inference
                if "pid" in infer_result and infer_result["pid"] != "N/A":
                    final_infer_results_map[infer_result["pid"]] = infer_result 
                
                # Periodically save results for robustness
                if (i + 1) % args.save_frequency == 0:
                    try:
                        # Save all results from the map
                        Utils.write_to_jsonl(list(final_infer_results_map.values()), save_path)
                        logging.info(f"Saved {len(final_infer_results_map)} inference results to {save_path} (after processing {i + 1} new samples).")
                    except Exception as save_e:
                        logging.error(f"Error saving results at iteration {i + 1}: {save_e}")
                
            except Exception as e:
                logging.error(f"Error processing inference for a sample: {e}")

    # Final save of all results after loop completion
    try:
        Utils.write_to_jsonl(list(final_infer_results_map.values()), save_path)
        logging.info(f"All {len(final_infer_results_map)} inference results saved to {save_path}.")
    except Exception as final_save_e:
        logging.error(f"Final save failed: {final_save_e}")

    print("-" * 50)
    print("Inference Generation Complete!")
    print(f"Number of samples newly processed (inferred by LLM): {shared_counts['processed_sample_num']}")
    print(f"Number of failed inferences (sample level errors or no LLM response): {shared_counts['fail_sample_num']}")
    print("-" * 50)
    logging.info("Inference generation process finished successfully.")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Script for generating LLM inference responses."
    )
    parser.add_argument("--model_name", type=str, default="Qwen3-235B-A22B_thinking",
                        help="Name of the model to use for inference (used for result directory and LLM initialization).")
    parser.add_argument("--save_frequency", type=int, default=1,
                        help="Frequency (in samples processed) to save intermediate results.")
    parser.add_argument("--infer_proc", type=int, default=20,
                        help="Number of parallel processes/threads to use for inference.")
    # pid_list is now a string to support comma-separated input from command line
    parser.add_argument("--pid_list", type=str, default="",
                        help="Comma-separated list of PIDs to specifically infer (e.g., '0,1,2'). Leave empty to infer all PIDs in the dataset.")
    parser.add_argument("--dataset_name", type=str, default="default_dataset",
                        help="Name of the dataset to load for inference.")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (e.g., 'train', 'validation', 'test').")

    # Add ModelArgsConfig parameters to the same parser
    parser = ModelArgsConfig.get_arg_parser(parser)
    
    # Parse all command-line arguments
    args = parser.parse_args()

    # Post-process pid_list string into a list of integers
    if args.pid_list:
        try:
            args.pid_list = [int(p.strip()) for p in args.pid_list.split(',')]
            logging.info(f"Parsed pid_list: {args.pid_list}")
        except ValueError:
            logging.error(f"Invalid pid_list format: {args.pid_list}. Please use comma-separated integers (e.g., '0,1,2').")
            sys.exit(1)
    else:
        args.pid_list = [] # If no input, an empty list means infer all

    # Configure logging
    log_filename = "inference.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[INFER] %(asctime)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="a" # Append to log file
    )
    # Add a stream handler to also print logs to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[INFER] %(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Running inference script: {__file__}")
    logging.info(f"Generating responses for model: {args.model_name}")
    logging.info(f"Logging output to: {log_filename}")

    # Call the main inference function
    try:
        infer_main(args)
    except Exception as e:
        logging.critical(f"An unhandled error occurred during inference: {e}", exc_info=True)
        sys.exit(1)