# T4FIDS (AI4FIDS)
This repository contains T4FIDS, which performs real federated training that can be deployed across various devices. More specifically, T4FIDS trains federated intrusion detection models that can later be used to detect malicious activities in the underlying infrastructure. T4FIDS expects tabular network flow statistics as input but is agnostic to the method used to generate these flows. It can be easily configured to support different types of flow data. T4FIDS uses the Flower framework to facilitate federated training.

## Installation

1. Clone the repo:
    ```shell
    git clone https://github.com/Metamind-Innovations/t4fids.git
    ```

2. Create a Python environment (make sure the `python-venv` package is already installed):
    ```shell
    python3 -m venv venv
    ```

3. Activate the environment and install the Python packages:
    ```shell
    cd t4fids
    source ./venv/bin/activate
    (venv) pip install -r requirements.txt
    ```

## Dataset setup

Place the train and test sets that the client will use in the ```data/``` directory as ```train.csv``` and ```test.csv```, respectively.
```
├── data/
│   ├── train.csv
│   ├── test.csv
```


## Configuration

The client and server configuration files should be located at:

```
├── conf/
│   ├── client/
        ├── config.json
│   ├── server/
        ├── config.json
```

A client configuration example comes as follows:

```json
DEFAULT_DASH_CONF = {
    "paths": {
        "data": "data",
        "model_save": "models",
        "results_save": "results"
    },
    "server": {
        "address": "127.0.0.1",
        "port": "8080"
    },
    "tr_params": {
        "learning_rate": 0.1,
        "batch_size": 64,
        "local_epochs": 5
    },
    "data_kw": {
        "features_to_drop": [
            "flow_id",
            "ip_src",
            "ip_dst",
            "udp_sport",
            "udp_dport"
        ],
        "label_keyword": "label"
    },
    "misc": {
        "override_params": false,
        "resample_flag": false
    },
    "opt": {
        "optimizer": "RMSprop",
        "params": {
            "rho": 0.9,
            "momentum": 0.0
        }
    }
}
```

while an example for the server is:

```json
{
    "server_address": "0.0.0.0",
    "server_port": "8080",
    "num_rounds": 20,
    "num_clients": 3,
    "batch_size": 32,
    "local_epochs": 1,
    "input_shape": 49,
    "num_classes": 5,
    "strategy": "FedAvg"
}
```

## Execution

To run the client (with the deafult configuration file in the respective path), execute:

```shell
python -m  scripts.run_client
```

To run the server (with the deafult configuration file in the respective path), execute:

```shell
python -m scripts.run_server
```

You can also configure the server via CLI, as per the following example:

```shell
python -m scripts.run_server --num_rounds 50 --batch_size 64 --strategy 'FedAdam'
```

To run the experiment successfully, the server and at least three client instances must be launched on separate devices, either physical or virtual.

## Results

After succesfully running the experiments you should see the global model, scaler, and label encoder, under the ```models/``` directory as:

```
├── models/
│   ├── model.h5
│   ├── scaler.joblib
│   ├── label_encoder.joblib
```

A suffix with the timestamp also appears in the name of the above artifacts. 

Finally, some training results with key evaluation metrics will appear in ```results/``` as ```metrics.json```.

## Execution with dashboard
To use the client dashboard as an interface for training, execute from top-level directory:
```shell
streamlit run t4fids/client/dashboard.py
```
You will then view the client app in the browser by using the relevant URL that appears.

## Run with Docker

To build the docker image of the client, run from the top-level directory

```shell
docker build -t t4fids-client -f docker/client/Dockerfile .
```
Similarly, to build the server's docker image, simply execute
```shell
docker build -t t4fids-server -f docker/server/Dockerfile .
```
Before running the client as a Docker container, you should create the following directory structure:

```
client/
├── config.json 
├── data/
│   ├── train.csv
│   ├── test.csv            
├── models/           
└── results/ 
```
This structure is required to enable volume mounting during container execution. Finally, you can run the client Docker container as follows:

```shell
docker run --rm \
  -v $(pwd)/client/config.json:/app/config.json \
  -v $(pwd)/client/data:/app/data \
  -v $(pwd)/client/models:/app/models \
  -v $(pwd)/client/results:/app/results \
  --name client \
  t4fids-client
```
Before running the server Docker image, create the ```server/``` directory and locate the server's configuration file, as ```config.json```. To run the server Docker container, execute:
```shell
PORT=$(jq -r '.server_port' server/config.json)    # Use a variable for port

docker run --rm \
  --name server \
  -v $(pwd)/server/config.json:/app/config.json \
  -p ${PORT}:${PORT} \
  t4fids-server
```
