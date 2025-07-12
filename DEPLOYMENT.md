# Deploying the Distributed Assessor System

This document outlines how to configure and deploy the distributed assessor system components.

## System Architecture Overview

The system consists of the following main components:

1.  **Redis Server**: Acts as the central task queue and result store. This must be deployed and accessible to the Master Server and all Assessor Workers.
2.  **Master Server (`assessor_server.py`)**:
    *   Receives assessment requests via gRPC from clients (Actor Workers).
    *   Enqueues these requests as tasks into the Redis queue.
    *   Typically, a single instance of the Master Server is run.
3.  **Assessor Workers (`assessor_worker.py`)**:
    *   Fetch tasks from the Redis queue.
    *   Perform the assessment (computation, typically GPU-bound).
    *   Store results back into Redis.
    *   Multiple instances of Assessor Workers can run concurrently, ideally one per available GPU, distributed across multiple machines.
4.  **Clients (`client.py` or integrated Actor Worker logic)**:
    *   Send assessment requests to the Master Server.
    *   Receive a `task_id` from the Master.
    *   Poll Redis using the `task_id` to retrieve the assessment result.

## Configuration via Environment Variables

The components are configured using environment variables.

### 1. Master Server (`assessor_server.py`)

*   `REDIS_HOST`: Hostname or IP address of the Redis server.
    *   Default: `127.0.0.1`
*   `REDIS_PORT`: Port number of the Redis server.
    *   Default: `6379`
*   `GRPC_PORT`: Port number for the Master Server to listen for gRPC requests on.
    *   Default: `50051`

**Example:**
```bash
export REDIS_HOST=10.0.0.5
export REDIS_PORT=6379
export GRPC_PORT=50051
python assessor_server.py
```

### 2. Assessor Worker (`assessor_worker.py`)

*   `REDIS_HOST`: Hostname or IP address of the Redis server.
    *   Default: `127.0.0.1`
*   `REDIS_PORT`: Port number of the Redis server.
    *   Default: `6379`
*   `CUDA_VISIBLE_DEVICES`: (Standard NVIDIA environment variable) Specifies which GPU this worker instance should use. This should be set uniquely for each worker process running on a multi-GPU machine.
    *   Example: `CUDA_VISIBLE_DEVICES=0 python assessor_worker.py` for GPU 0, `CUDA_VISIBLE_DEVICES=1 python assessor_worker.py` for GPU 1.

**Example:**
```bash
export REDIS_HOST=10.0.0.5
export REDIS_PORT=6379
CUDA_VISIBLE_DEVICES=0 python assessor_worker.py &
CUDA_VISIBLE_DEVICES=1 python assessor_worker.py &
# ... for each GPU
```

### 3. Client (`client.py` or Actor Integration)

*   `MASTER_HOST`: Hostname or IP address of the Master Server.
    *   Default: `127.0.0.1`
*   `MASTER_PORT`: Port number the Master Server is listening on.
    *   Default: `50051`
*   `REDIS_HOST`: Hostname or IP address of the Redis server (for polling results).
    *   Default: `127.0.0.1`
*   `REDIS_PORT`: Port number of the Redis server (for polling results).
    *   Default: `6379`

**Example:**
```bash
export MASTER_HOST=10.0.0.10 # IP of the machine running assessor_server.py
export MASTER_PORT=50051
export REDIS_HOST=10.0.0.5   # IP of the machine running Redis
export REDIS_PORT=6379
python client.py
```

## Deployment Notes

*   Ensure network connectivity between all components:
    *   Clients must be able to reach the Master Server's gRPC port.
    *   Master Server and all Assessor Workers must be able to reach the Redis server.
    *   Clients must be able to reach the Redis server for polling results.
*   For GPU utilization, each `assessor_worker.py` instance should be pinned to a specific GPU using `CUDA_VISIBLE_DEVICES`. The number of worker instances on an Assessor machine should typically match the number of available GPUs.
*   Consider using a process manager (like `systemd`, `supervisor`, or container orchestration like Docker/Kubernetes) to manage the Master Server and Assessor Worker processes for robustness and scalability in a production environment.
