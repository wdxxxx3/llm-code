print("DEBUG: assessor_server.py execution started")
import sys
sys.stdout.flush()

import grpc
from concurrent import futures
import time
import json
import uuid
import redis

import assessor_pb2
import assessor_pb2_grpc
import os

# Configuration from environment variables with defaults
REDIS_HOST = os.environ.get('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
GRPC_PORT = int(os.environ.get('GRPC_PORT', '50051'))
REDIS_TASK_QUEUE_NAME = 'assessor_task_queue' # This could also be configurable if needed

class AssessorServicer(assessor_pb2_grpc.AssessorServiceServicer):
    def __init__(self):
        self.redis_host = REDIS_HOST
        self.redis_port = REDIS_PORT
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=0, decode_responses=False)
            self.redis_client.ping() # Verify connection
            print(f"[Server] Connected to Redis at {self.redis_host}:{self.redis_port}")
        except redis.exceptions.ConnectionError as e:
            print(f"[Server] CRITICAL: Could not connect to Redis at {self.redis_host}:{self.redis_port}. Error: {e}")
            self.redis_client = None


    def ProcessAssessment(self, request, context):
        if not self.redis_client:
            print("[Server] Error: No Redis connection available.")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error: No Redis connection.")
            return assessor_pb2.AssessorResponse(
                request_id=request.request_id, # Return original client request_id
                status="ERROR",
                error_message="Failed to connect to task queue backend."
            )

        task_id = str(uuid.uuid4())
        print(f"[Server] Received assessment request: {request.request_id}. Assigning task_id: {task_id}")

        task_payload = {
            "task_id": task_id, # Master generated task_id
            "client_request_id": request.request_id, # Original request_id from client for later matching
            "source_machine": request.source_machine,
            "source_worker_id": request.source_worker_id,
            "payload": request.payload.decode('utf-8'), # Assuming payload is text-based like JSON, decode here
                                                       # If binary, consider base64 encoding or keeping as bytes if supported
            "gpu_id": request.gpu_id # Still passing this along, worker can decide to use/ignore
        }

        try:
            # Serialize the task payload to JSON string
            task_json = json.dumps(task_payload)

            # Push the task to the Redis queue
            self.redis_client.lpush(REDIS_TASK_QUEUE_NAME, task_json)
            print(f"[Server] Task {task_id} (client_request_id: {request.request_id}) enqueued into '{REDIS_TASK_QUEUE_NAME}'.")

            # Return response to client indicating task is queued
            return assessor_pb2.AssessorResponse(
                request_id=task_id, # Return the Master-generated task_id
                status="QUEUED",
                result=b"", # No direct result at this point
                error_message=""
            )
        except redis.exceptions.ConnectionError as e:
            print(f"[Server] Error communicating with Redis while enqueuing task {task_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error: Failed to enqueue task.")
            return assessor_pb2.AssessorResponse(
                request_id=request.request_id, # Return original client request_id
                status="ERROR",
                error_message="Failed to enqueue task due to backend communication error."
            )
        except Exception as e:
            print(f"[Server] Error processing request {request.request_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return assessor_pb2.AssessorResponse(
                request_id=request.request_id, # Return original client request_id
                status="ERROR",
                error_message=f"An unexpected error occurred: {str(e)}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    assessor_pb2_grpc.add_AssessorServiceServicer_to_server(AssessorServicer(), server)
    grpc_listen_address = f'[::]:{GRPC_PORT}'
    server.add_insecure_port(grpc_listen_address)
    print(f"[Server] Starting Master gRPC server on {grpc_listen_address}, queueing to Redis list '{REDIS_TASK_QUEUE_NAME}' at {REDIS_HOST}:{REDIS_PORT}...")
    server.start()
    try:
        while True:
            time.sleep(86400)  # One day
    except KeyboardInterrupt:
        print("[Server] Stopping Master server...")
        server.stop(0)
        print("[Server] Master server stopped.")

if __name__ == '__main__':
    serve()
