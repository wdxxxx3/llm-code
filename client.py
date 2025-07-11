import grpc
import assessor_pb2
import assessor_pb2_grpc
import uuid
import json

def run_client(gpu_id_to_use=0):
    # Establish a channel to the server
    # For this prototype, the server is running on localhost
    with grpc.insecure_channel('localhost:50051') as channel:
        # Create a stub (client)
        stub = assessor_pb2_grpc.AssessorServiceStub(channel)

        # Create a request message
        request_id = str(uuid.uuid4())
        sample_payload_data = {"model_name": "test_model_v1", "input_features": [1.0, 2.5, 3.0]}

        request = assessor_pb2.AssessorRequest(
            request_id=request_id,
            source_machine="actor_machine_1",
            source_worker_id="worker_process_alpha",
            payload=json.dumps(sample_payload_data).encode('utf-8'), # Example payload as JSON string
            gpu_id=gpu_id_to_use
        )

        print(f"[Client] Sending request {request.request_id} to GPU {request.gpu_id} with payload: {sample_payload_data}")

        try:
            # Make the RPC call
            response = stub.ProcessAssessment(request, timeout=40) # Set a timeout for the call

            print(f"\n[Client] Received response for request {response.request_id}:")
            print(f"  Status: {response.status}")
            if response.result:
                try:
                    result_data = json.loads(response.result.decode('utf-8'))
                    print(f"  Result: {result_data}")
                except json.JSONDecodeError:
                    print(f"  Result (raw bytes): {response.result}")
            if response.error_message:
                print(f"  Error: {response.error_message}")

        except grpc.RpcError as e:
            print(f"[Client] RPC failed: {e.code()} - {e.details()}")
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print("[Client] The server took too long to respond.")
        except Exception as e:
            print(f"[Client] An unexpected error occurred: {e}")

if __name__ == '__main__':
    print("--- Running client for GPU 0 ---")
    run_client(gpu_id_to_use=0)
    # print("\n--- Running client for GPU 1 (simulated) ---")
    # run_client(gpu_id_to_use=1) # Example for testing with another GPU ID
    # print("\n--- Running client for GPU 7 (simulated) ---")
    # run_client(gpu_id_to_use=7) # Example for testing with another GPU ID
