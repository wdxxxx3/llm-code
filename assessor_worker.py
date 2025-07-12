import redis
import json
import time
import os
import sys

# Configuration from environment variables with defaults
REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_TASK_QUEUE_NAME = 'assessor_task_queue' # Could also be configurable
REDIS_RESULT_KEY_PREFIX = 'result:' # Could also be configurable
REDIS_RESULT_EXPIRY_SECONDS = 3600 # 1 hour

# Worker function to simulate GPU task (adapted from the original server)
def perform_assessment(task_details):
    """
    Simulates a task running on a specific GPU.
    """
    gpu_id = task_details.get('gpu_id', 'any') # Default to 'any' if not specified
    task_id = task_details['task_id']
    client_request_id = task_details['client_request_id']
    payload_str = task_details['payload'] # Already decoded by server

    # Simulate setting CUDA_VISIBLE_DEVICES for the worker process
    # In a real scenario, this would ensure the process uses only the specified GPU.
    # For this prototype, we're just acknowledging the gpu_id.
    # If multiple workers run on the same machine, each needs a distinct GPU.
    # This script assumes it's launched with CUDA_VISIBLE_DEVICES already set externally
    # OR it's selecting a GPU based on 'gpu_id' if that's the design.
    # For simplicity here, we'll just print it.
    # If this worker is meant to be pinned to a specific GPU, that should be handled
    # when launching this script (e.g. CUDA_VISIBLE_DEVICES=0 python assessor_worker.py)

    actual_gpu_for_worker = os.environ.get("CUDA_VISIBLE_DEVICES", f"simulated_gpu_{gpu_id}")

    print(f"[Worker PID: {os.getpid()}, GPU: {actual_gpu_for_worker}] Processing task_id: {task_id} (client_req: {client_request_id}).")

    # Simulate work based on payload
    try:
        # Assuming payload was originally JSON, as per server's decode
        payload_data = json.loads(payload_str)
        # Example: work time could depend on payload content
        work_time = len(payload_data.get("input_features", [])) if isinstance(payload_data, dict) else 2
        time.sleep(max(1, min(work_time, 10))) # Simulate work for 1-10 seconds
    except json.JSONDecodeError:
        print(f"[Worker PID: {os.getpid()}] Warning: Payload for task {task_id} was not valid JSON. Using default work time.")
        time.sleep(5)
    except Exception as e:
        print(f"[Worker PID: {os.getpid()}] Error during simulated work for task {task_id}: {e}. Using default work time.")
        time.sleep(5)


    # Simulate result
    result_data = {
        "score": round(time.time() % 1.0, 3), # Some dynamic score
        "message": f"Assessment complete for task {task_id} by worker {os.getpid()} on GPU {actual_gpu_for_worker}",
        "original_payload_preview": payload_str[:100] + "..." if len(payload_str) > 100 else payload_str,
        "processed_by_pid": os.getpid()
    }
    print(f"[Worker PID: {os.getpid()}, GPU: {actual_gpu_for_worker}] Work finished for task_id: {task_id}.")
    return result_data

def main_worker_loop():
    print(f"[Worker PID: {os.getpid()}] Starting up...")
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        redis_client.ping()
        print(f"[Worker PID: {os.getpid()}] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.exceptions.ConnectionError as e:
        print(f"[Worker PID: {os.getpid()}] CRITICAL: Could not connect to Redis. Error: {e}. Exiting.")
        sys.exit(1)

    print(f"[Worker PID: {os.getpid()}] Waiting for tasks from queue '{REDIS_TASK_QUEUE_NAME}'...")
    while True:
        try:
            # Blocking pop from the right of the list (tasks are LPUSHed by server)
            # Returns a tuple: (queue_name, task_json_bytes) or None if timeout
            message = redis_client.brpop(REDIS_TASK_QUEUE_NAME, timeout=0) # 0 = block indefinitely

            if message:
                _queue_name, task_json_bytes = message
                task_details = json.loads(task_json_bytes.decode('utf-8'))
                task_id = task_details['task_id']

                print(f"[Worker PID: {os.getpid()}] Received task: {task_id}")

                try:
                    assessment_result = perform_assessment(task_details)
                    result_key = f"{REDIS_RESULT_KEY_PREFIX}{task_id}"
                    result_json = json.dumps(assessment_result)

                    redis_client.setex(name=result_key, time=REDIS_RESULT_EXPIRY_SECONDS, value=result_json)
                    print(f"[Worker PID: {os.getpid()}] Result for task {task_id} stored in Redis at '{result_key}'.")

                except Exception as e:
                    print(f"[Worker PID: {os.getpid()}] Error processing task {task_id}: {e}")
                    # Store an error result perhaps?
                    result_key = f"{REDIS_RESULT_KEY_PREFIX}{task_id}"
                    error_result = {
                        "error": str(e),
                        "message": "Worker failed to process task.",
                        "task_details": task_details # Include original task for debugging
                    }
                    redis_client.setex(name=result_key, time=REDIS_RESULT_EXPIRY_SECONDS, value=json.dumps(error_result))
                    print(f"[Worker PID: {os.getpid()}] Error result for task {task_id} stored in Redis.")

            # Brief sleep if brpop times out (if timeout > 0) or just to yield, though brpop is blocking.
            # time.sleep(0.01)

        except redis.exceptions.ConnectionError as e:
            print(f"[Worker PID: {os.getpid()}] Redis connection error: {e}. Attempting to reconnect...")
            time.sleep(5) # Wait before retrying connection
            try:
                redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
                redis_client.ping()
                print(f"[Worker PID: {os.getpid()}] Reconnected to Redis.")
            except redis.exceptions.ConnectionError:
                print(f"[Worker PID: {os.getpid()}] Failed to reconnect to Redis. Will retry.")
        except KeyboardInterrupt:
            print(f"[Worker PID: {os.getpid()}] Shutting down...")
            break
        except Exception as e:
            print(f"[Worker PID: {os.getpid()}] Unexpected error in main loop: {e}")
            time.sleep(5) # Avoid rapid spin-fail

if __name__ == '__main__':
    # To simulate multiple workers on different GPUs on the same machine,
    # you would typically launch this script multiple times, setting
    # CUDA_VISIBLE_DEVICES for each launch. For example:
    # CUDA_VISIBLE_DEVICES=0 python assessor_worker.py &
    # CUDA_VISIBLE_DEVICES=1 python assessor_worker.py &
    # ...
    main_worker_loop()
