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

Place the train and test sets in the ```data/``` directory as ```train.csv``` and ```test.csv```, respectively.
<pre>
├── data/
│   ├── train.csv
│   ├── test.csv
</pre> 

## Configuration

The client and server configuration files should be located at:

<pre>
├── conf/
│   ├── client/
        ├── config.json
│   ├── server/
        ├── config.json
</pre> 

A client configuration example comes as follows:

<pre>
{
    "data_path": "data",
    "server_address": "127.0.0.1",
    "server_port": "8080",
    "model_save_path": "models",
    "results_save_path": "results",
    "learning_rate": 0.002,
    "resample_flag": false,
    "features_to_drop": ["flow_id","src_ip","dst_ip","src_port", "dst_port"],
    "label_keyword": "label"
}
</pre> 

while an example for the server is:

<pre>
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
</pre>

## Execution

To run the client (with the deafult configuration file in the respective path), execute:

```shell
python -m  scripts.run_client
```

To run the server (with the deafult configuration file in the respective path), execute:

```shell
python -m scripts.run_server.py
```

You can also configure the server via CLI, as per the following example:

```shell
python -m scripts.run_server.py --num_rounds 50 --batch_size 64 --strategy 'FedAdam'
```

To run the experiment successfully, the server and at least three client instances must be launched on separate devices, either physical or virtual.

## Results

After succesfully running the experiments you should see the global model, scaler, and label encoder, under the ```models/``` directory as:

<pre>
├── models/
│   ├── model.h5
│   ├── scaler.joblib
    ├── label_encoder.joblib
</pre> 

A suffix with the timestamp also appears in the name of the above artifacts. 

Finally, some training results with key evaluation metrics will appear in ```results/``` as ```metrics.json```.