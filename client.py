import grpc
import assessor_pb2
import assessor_pb2_grpc
import uuid
import json
import time
import redis
import os

# Configuration from environment variables with defaults
REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
MASTER_HOST = os.environ.get('MASTER_HOST', '127.0.0.1') # Assuming master runs on localhost by default for client
MASTER_PORT = int(os.environ.get('MASTER_PORT', '50051'))

REDIS_RESULT_KEY_PREFIX = 'result:' # Could also be configurable

# Polling configuration
POLL_INTERVAL_SECONDS = 0.5
MAX_POLL_ATTEMPTS = 120 # e.g., 120 attempts * 0.5s/attempt = 60 seconds timeout

def run_client(gpu_id_to_use=0):
    master_address = f"{MASTER_HOST}:{MASTER_PORT}"
    # Establish a channel to the gRPC server (Master)
    try:
        with grpc.insecure_channel(master_address) as channel:
            stub = assessor_pb2_grpc.AssessorServiceStub(channel)
            client_request_id = str(uuid.uuid4()) # This is the ID the client initially tracks
            sample_payload_data = {"model_name": "test_model_v2_redis", "input_features": [1.5, 2.0, 3.5, 4.0]}

            request = assessor_pb2.AssessorRequest(
                request_id=client_request_id,
                source_machine="actor_machine_kube_1",
                source_worker_id="worker_process_beta",
                payload=json.dumps(sample_payload_data).encode('utf-8'),
                gpu_id=gpu_id_to_use
            )

            print(f"[Client] Sending gRPC request (client_request_id: {client_request_id}) to Master for GPU {request.gpu_id} with payload: {sample_payload_data}")

            # Make the RPC call to the Master
            master_response = stub.ProcessAssessment(request, timeout=10) # Timeout for gRPC call itself

            if master_response.status == "QUEUED" and master_response.request_id: # master_response.request_id is the task_id
                task_id = master_response.request_id
                print(f"[Client] Task enqueued by Master. Task ID: {task_id} (Original client_request_id: {client_request_id})")
                print(f"[Client] Now polling Redis for result at key '{REDIS_RESULT_KEY_PREFIX}{task_id}'...")

                # Connect to Redis to poll for results
                try:
                    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
                    redis_client.ping()
                except redis.exceptions.ConnectionError as e:
                    print(f"[Client] Error: Could not connect to Redis to poll for results: {e}")
                    return

                for attempt in range(MAX_POLL_ATTEMPTS):
                    result_json = redis_client.get(f"{REDIS_RESULT_KEY_PREFIX}{task_id}")
                    if result_json:
                        print(f"\n[Client] Result found for Task ID {task_id} (attempt {attempt + 1}):")
                        try:
                            result_data = json.loads(result_json)
                            print(f"  Status: COMPLETED (from worker)")
                            print(f"  Result: {result_data}")
                            # Optionally, client could delete the result key from Redis if it's meant to be consumed once
                            # redis_client.delete(f"{REDIS_RESULT_KEY_PREFIX}{task_id}")
                        except json.JSONDecodeError:
                            print(f"  Error: Could not decode result JSON: {result_json}")
                        return # Exit after finding result

                    # print(f"[Client] Attempt {attempt + 1}/{MAX_POLL_ATTEMPTS}: Result not found yet for task {task_id}. Waiting {POLL_INTERVAL_SECONDS}s...")
                    time.sleep(POLL_INTERVAL_SECONDS)

                print(f"[Client] Polling timeout for Task ID {task_id} after {MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS} seconds.")

            elif master_response.status == "ERROR":
                print(f"\n[Client] Master returned an error for client_request_id {client_request_id}:")
                print(f"  Master Task ID (if any): {master_response.request_id}")
                print(f"  Error Message: {master_response.error_message}")
            else:
                print(f"\n[Client] Unexpected response from Master for client_request_id {client_request_id}:")
                print(f"  Task ID: {master_response.request_id}")
                print(f"  Status: {master_response.status}")
                print(f"  Error: {master_response.error_message}")
                print(f"  Result (bytes): {master_response.result}")


    except grpc.RpcError as e:
        print(f"[Client] gRPC RPC failed: {e.code()} - {e.details()}")
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            print("[Client] The Master server took too long to respond to the gRPC request.")
    except Exception as e:
        print(f"[Client] An unexpected error occurred: {e}")

if __name__ == '__main__':
    print("--- Running client (will send request to Master and poll Redis for GPU 0) ---")
    run_client(gpu_id_to_use=0)

    # Example for sending another request, perhaps targeting a different simulated GPU
    # print("\n--- Running client (will send request to Master and poll Redis for GPU 1) ---")
    # run_client(gpu_id_to_use=1)
