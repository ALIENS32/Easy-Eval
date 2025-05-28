import argparse
import concurrent
from concurrent.futures import ThreadPoolExecutor
import logging
import tqdm
from config.model_args_config import ModelArgsConfig
import os
import sys
from dataclasses import asdict
import threading # Import threading module for thread safety

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
                continue # Continue to the next attempt if the key is missing
        except Exception as e:
            logging.warning(f"API Retry: Attempt {attempt + 1}/10 failed with error: {e}")
    logging.error(f"Failed to get response after 10 attempts for query: {query}")
    return ""

# Define a lock for protecting shared counters
shared_counts_lock = threading.Lock()

def evaluate_sample(llm, cur_eval_object, cur_infer_result, dataset_dict, shared_counts):
    """
    Evaluates a single sample.
    ! Note: The actual evaluation logic needs to be implemented here.
    Updates shared_counts in a thread-safe manner.
    """
    try:
        # Increment evaluated sample count (thread-safe)
        with shared_counts_lock:
            shared_counts["evaluated_sample_num"] += 1
        
        logging.debug(f"Evaluating sample PID: {cur_eval_object['pid']}")
        
        # --- PLACE YOUR ACTUAL EVALUATION LOGIC HERE ---
        # Example of how you might use the LLM for evaluation:
        # evaluator_query = DataTemplate.get_evaluator_query(cur_infer_result, dataset_dict.get(cur_eval_object['pid']))
        # evaluator_response = get_answer(llm, evaluator_query) # Re-use get_answer for robustness
        #
        # if "correct" in evaluator_response.lower(): # Example check
        #     cur_eval_object["status"] = "PASS"
        # else:
        #     cur_eval_object["status"] = "FAIL"
        #     with shared_counts_lock: # Protect shared counter
        #         shared_counts["fail_sample_num"] += 1
        # --- END OF EVALUATION LOGIC PLACEHOLDER ---

        # If your evaluation involves further API calls, you might update these:
        # with shared_counts_lock:
        #     shared_counts["total_evaluated"] += 1
        #     if evaluation_api_call_failed:
        #         shared_counts["failed_evaluations"] += 1

        return cur_eval_object
    except Exception as e:
        logging.error(f"Error evaluating sample PID {cur_eval_object.get('pid', 'N/A')}: {e}")
        # Increment fail sample count (thread-safe)
        with shared_counts_lock:
            shared_counts["fail_sample_num"] += 1
        cur_eval_object["status"] = "ERROR" # Mark as error
        return cur_eval_object

def eval_main(args):
    """
    Main function for orchestrating the evaluation process.
    Handles LLM initialization, data loading, parallel evaluation, and result saving.
    """
    logging.info("Starting evaluation process.")

    # Create ModelArgsConfig instance from parsed arguments, using 'evaluator' for the LLM model name
    model_args = asdict(ModelArgsConfig.from_argparse_args(args))
    try:
        llm = LLM(args.evaluator,model_args)
        logging.info(f"LLM initialized with model arguments: {model_args}")
    except Exception as e:
        logging.critical(f"Failed to initialize LLM: {e}")
        sys.exit(1) # Exit if LLM cannot be initialized

    # Define paths for storing results using object_model_name for consistency
    infer_path = os.path.join("results", args.object_model_name, f"infer_result.jsonl")
    save_path = os.path.join("results", args.object_model_name, "eval_result.jsonl")

    # Create directories if they don't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        logging.info(f"Created directory: {os.path.dirname(save_path)}")
    else:
        logging.info(f"Directory already exists: {os.path.dirname(save_path)}")

    # Load all inference results and existing evaluation results
    all_infer_results = Utils.read_from_jsonl(infer_path)
    all_existing_eval_results = Utils.read_from_jsonl(save_path)
    dataset_dict = Utils.load_data(args.dataset_name, args.split)

    # Filter inference results based on provided pid_list
    infer_results_to_process = []
    if args.pid_list:
        logging.info(f"Filtering inference results to process only PIDs: {args.pid_list}")
        for res in all_infer_results:
            if res["pid"] in args.pid_list:
                infer_results_to_process.append(res)
        if len(infer_results_to_process) != len(args.pid_list):
            logging.warning(f"Some PIDs from pid_list ({len(args.pid_list)}) were not found in infer_results ({len(infer_results_to_process)} found).")
    else:
        logging.info("No pid_list specified, processing all inference results.")
        infer_results_to_process = all_infer_results

    # Create a mapping from PID to inference result for quick lookup
    pid_infer_dict = {res["pid"]: res for res in infer_results_to_process}  
    
    # Initialize evaluation objects. If existing_eval_results are present, load them.
    # Otherwise, create templates for new inference results.
    existing_eval_dict = {res["pid"]: res for res in all_existing_eval_results}
    
    # Store all evaluation results (both historical and new) in a dictionary for easy updates and final saving
    final_eval_results_map = {res["pid"]: res for res in all_existing_eval_results}

    samples_to_evaluate = []
    for infer_res in infer_results_to_process:
        pid = infer_res["pid"]
        # Check if this PID has already been evaluated and completed
        # Assuming DataTemplate.get_eval_result_template initializes status to 'PENDING' or similar
        if pid in final_eval_results_map and final_eval_results_map[pid].get("status") not in ["PENDING", None, ""]: 
             logging.debug(f"PID {pid} already evaluated with status '{final_eval_results_map[pid].get('status')}', skipping.")
        else:
            # If not evaluated or status is pending, create a new template or use existing partial one
            if pid in existing_eval_dict:
                eval_obj = existing_eval_dict[pid]
            else:
                eval_obj = DataTemplate.get_eval_result_template(pid)
            samples_to_evaluate.append(eval_obj)

    logging.info(f"Will evaluate {len(samples_to_evaluate)} samples out of {len(infer_results_to_process)} samples relevant to pid_list.")
    if all_existing_eval_results:
        logging.info("Incorporated existing results where available for resumed evaluation.")

    # Initialize shared evaluation counters
    shared_counts = {
        "evaluated_sample_num": 0,
        "fail_sample_num": 0,
        "total_evaluated": 0,
        "failed_evaluations": 0
    }
    logging.info("Shared counters initialized.")

    def worker(cur_eval_object):
        """Worker function for the thread pool to evaluate a single sample."""
        pid = cur_eval_object["pid"]
        cur_infer_result = pid_infer_dict.get(pid)
        if not cur_infer_result:
            logging.warning(f"Inference result not found for PID: {pid}. Skipping evaluation for this sample.")
            return cur_eval_object # Returns the unchanged eval_object, effectively skipping evaluation
        
        # Evaluate the sample and update counts
        eval_result = evaluate_sample(llm, cur_eval_object, cur_infer_result, dataset_dict, shared_counts)
        return eval_result

    # Process samples using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.infer_proc) as executor:
        futures = [executor.submit(worker, eval_object) for eval_object in samples_to_evaluate]
        logging.info(f"Submitted {len(futures)} evaluation tasks to the thread pool with {args.infer_proc} workers.")
        
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating samples")):
            try:
                eval_result = future.result()
                # Update the final map with the result of the current evaluation
                final_eval_results_map[eval_result["pid"]] = eval_result 
                
                # Periodically save results for robustness
                if (i + 1) % args.save_frequency == 0:
                    try:
                        # Save all results from the map
                        Utils.write_to_jsonl(list(final_eval_results_map.values()), save_path)
                        logging.info(f"Saved {len(final_eval_results_map)} evaluation results to {save_path} (after processing {i + 1} new samples).")
                    except Exception as save_e:
                        logging.error(f"Error saving results at iteration {i + 1}: {save_e}")
                
            except Exception as e:
                logging.error(f"Error processing evaluation for a sample: {e}")

    # Final save of all results after loop completion
    try:
        Utils.write_to_jsonl(list(final_eval_results_map.values()), save_path)
        logging.info(f"All {len(final_eval_results_map)} evaluation results saved to {save_path}.")
    except Exception as final_save_e:
        logging.error(f"Final save failed: {final_save_e}")

    print("-" * 50)
    print("Evaluation Complete!")
    print(f"Number of samples newly processed (evaluated by LLM): {shared_counts['evaluated_sample_num']}")
    print(f"Number of failed evaluations (sample level errors): {shared_counts['fail_sample_num']}")
    print(f"Total API evaluations (if applicable within evaluate_sample): {shared_counts['total_evaluated']}")
    print(f"Failed API evaluations (if applicable within evaluate_sample): {shared_counts['failed_evaluations']}")
    print("-" * 50)
    logging.info("Evaluation process finished successfully.")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Script for evaluating LLM responses."
    )
    parser.add_argument("--object_model_name", type=str, default="Qwen3-235B-A22B_thinking",
                        help="Name of the model being evaluated (used for result directory).")
    parser.add_argument("--evaluator", type=str, default="o3-mini-high",
                        help="Name of the evaluator model/method (the LLM performing the evaluation).")
    parser.add_argument("--save_frequency", type=int, default=1,
                        help="Frequency (in samples processed) to save intermediate results.")
    parser.add_argument("--infer_proc", type=int, default=20,
                        help="Number of parallel processes/threads to use for inference/evaluation.")
    # pid_list is now a string to support comma-separated input from command line
    parser.add_argument("--pid_list", type=str, default="",
                        help="Comma-separated list of PIDs to specifically evaluate (e.g., '0,1,2'). Leave empty to evaluate all PIDs in the inference results.")
    parser.add_argument("--dataset_name", type=str, default="default_dataset",
                        help="Name of the dataset to load for evaluation.")
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
        args.pid_list = [] # If no input, an empty list means evaluate all

    # Configure logging
    log_filename = "premise_critique.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[EVAL] %(asctime)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="a" # Append to log file
    )
    # Add a stream handler to also print logs to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[EVAL] %(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Running evaluation script: {__file__}")
    # Log the object_model_name for clarity
    logging.info(f"Evaluating model: {args.object_model_name} using evaluator: {args.evaluator}")
    logging.info(f"Logging output to: {log_filename}")

    # Call the main evaluation function
    try:
        eval_main(args)
    except Exception as e:
        logging.critical(f"An unhandled error occurred during evaluation: {e}", exc_info=True)
        sys.exit(1)