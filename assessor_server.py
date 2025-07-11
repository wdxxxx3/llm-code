import grpc
from concurrent import futures
import multiprocessing
import time
import os
import json

import assessor_pb2
import assessor_pb2_grpc

# Worker function to simulate GPU task
def assessment_worker(request_dict, result_queue, gpu_id):
    """
    Simulates a task running on a specific GPU.
    For the prototype, CUDA_VISIBLE_DEVICES is set, but no actual GPU computation happens.
    """
    # Simulate setting CUDA_VISIBLE_DEVICES for the worker process
    # In a real scenario, this would ensure the process uses only the specified GPU.
    # For this prototype, we're just acknowledging the gpu_id.
    print(f"[Worker PID: {os.getpid()}] Assigned to GPU: {gpu_id}. Simulating work for request: {request_dict['request_id']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Simulate work
    time.sleep(5) # Simulate a 5-second task

    # Simulate result
    result_payload = {
        "score": 0.95,
        "message": f"Assessment complete for payload processed by simulated GPU {gpu_id}",
        "original_payload_size": len(request_dict['payload'])
    }

    result_queue.put({
        "request_id": request_dict['request_id'],
        "status": "COMPLETED",
        "result": json.dumps(result_payload).encode('utf-8'),
        "error_message": ""
    })
    print(f"[Worker PID: {os.getpid()}] Work finished for request: {request_dict['request_id']}")

class AssessorServicer(assessor_pb2_grpc.AssessorServiceServicer):
    def ProcessAssessment(self, request, context):
        print(f"[Server] Received assessment request: {request.request_id} for GPU: {request.gpu_id}")

        # Use a multiprocessing.Queue for communication between processes
        result_queue = multiprocessing.Queue()

        # Convert request to dict for easier passing to process (protobuf objects might not be directly picklable)
        request_dict = {
            "request_id": request.request_id,
            "source_machine": request.source_machine,
            "source_worker_id": request.source_worker_id,
            "payload": request.payload, # Keep as bytes
            "gpu_id": request.gpu_id
        }

        # Start a new process for the assessment task
        # Pass gpu_id to the worker function
        process = multiprocessing.Process(target=assessment_worker, args=(request_dict, result_queue, request.gpu_id))
        process.start()

        print(f"[Server] Started worker process {process.pid} for request {request.request_id} on GPU {request.gpu_id}")

        # Wait for the result from the worker process
        # This makes the server synchronous for this prototype step
        try:
            # Blocking get with a timeout to prevent indefinite hanging
            result_data = result_queue.get(timeout=30)
            process.join(timeout=5) # Ensure process terminates
            if process.is_alive():
                print(f"[Server] Worker process {process.pid} did not terminate, killing.")
                process.terminate()
                process.join()

            print(f"[Server] Received result from worker for request {request.request_id}")
            return assessor_pb2.AssessorResponse(
                request_id=result_data["request_id"],
                status=result_data["status"],
                result=result_data["result"],
                error_message=result_data["error_message"]
            )
        except multiprocessing.queues.Empty:
            print(f"[Server] Timeout waiting for result from worker for request {request.request_id}")
            if process.is_alive():
                process.terminate()
                process.join()
            return assessor_pb2.AssessorResponse(
                request_id=request.request_id,
                status="ERROR",
                error_message="Processing timeout in worker."
            )
        except Exception as e:
            print(f"[Server] Error processing request {request.request_id}: {e}")
            if process.is_alive():
                process.terminate()
                process.join()
            return assessor_pb2.AssessorResponse(
                request_id=request.request_id,
                status="ERROR",
                error_message=str(e)
            )

def serve():
    # Set multiprocessing start method to 'spawn' for consistency, especially on macOS/Windows
    # if it's not the default. 'fork' can have issues with threads and resources.
    # On Linux, 'fork' is often the default and usually fine.
    # However, for CUDA related work or libraries that manage global state, 'spawn' is safer.
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("[Server] Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # This might happen if it's already set or on platforms where it's not allowed to change after context is used.
        print("[Server] Could not set multiprocessing start method to 'spawn', using default.")


    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    assessor_pb2_grpc.add_AssessorServiceServicer_to_server(AssessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("[Server] Starting server on port 50051...")
    server.start()
    try:
        while True:
            time.sleep(86400)  # One day
    except KeyboardInterrupt:
        print("[Server] Stopping server...")
        server.stop(0)
        print("[Server] Server stopped.")

if __name__ == '__main__':
    serve()
