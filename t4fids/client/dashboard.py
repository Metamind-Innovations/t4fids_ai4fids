import streamlit as st
import json
import os
import time
import pandas as pd
import subprocess
import tensorflow as tf
from pathlib import Path

def wait_for_path(path, check_interval=0.2):
    while not os.path.exists(path):
        time.sleep(check_interval)
        
def remove_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def load_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    return config
        
def display_metrics(metrics_path):
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        metrics = tuple(m * 100 for m in metrics)
        subcol1, subcol2, subcol3, subcol4 = st.columns(4)
        subcol1.metric("Accuracy", f"{metrics[0]:.2f}%")
        subcol2.metric("TPR", f"{metrics[1]:.2f}%")
        subcol3.metric("FPR", f"{metrics[2]:.2f}%")
        subcol4.metric("F1 Score", f"{metrics[3]:.2f}%")

def get_feature_lst(path, label_kw):
    data = pd.read_csv(Path(path) / 'train.csv')
    data = data.drop([label_kw], axis=1)
    cols = list(data.columns)
    return cols

def display_optimizer_params(optimizer_name, disabled_flag):
    params = {}
    
    if optimizer_name in ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad"]:
        if optimizer_name == "RMSprop":
            params["rho"] = st.number_input("Rho", value=0.9, min_value=0.0, max_value=1.0, disabled=disabled_flag)
            params["momentum"] = st.number_input("Momentum", value=0.0, min_value=0.0, disabled=disabled_flag)
        elif optimizer_name == "Adam":
            params["beta_1"] = st.number_input("Beta 1", value=0.99, min_value=0.0, max_value=1.0, disabled=disabled_flag)
            params["beta_2"] = st.number_input("Beta 2", value=0.999, min_value=0.0, max_value=1.0, disabled=disabled_flag)
        elif optimizer_name == "Adadelta":
            params["rho"] = st.number_input("Rho", value=0.95, min_value=0.0, max_value=1.0, disabled=disabled_flag)
        elif optimizer_name == "SGD":
            params["momentum"] = st.number_input("Momentum", value=0.0, min_value=0.0, max_value=1.0, disabled=disabled_flag)
    
    return params

# List of available optimizers in TensorFlow
OPTIMIZERS = {
    "RMSprop": tf.keras.optimizers.RMSprop,
    "Adam": tf.keras.optimizers.Adam,
    "Adadelta": tf.keras.optimizers.Adadelta,
    "SGD": tf.keras.optimizers.SGD
}

# load config
CONFIG_PATH = 'conf/client/config_dashboard.json'
config = load_config(CONFIG_PATH)

# Set the page configuration
st.set_page_config(page_title="Client T4FIDS Dashboard", layout="wide")

#st.image("logo.png", width=300)  # Adjust the width as needed

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            color: #1f77b4;
            font-size: 3em;
            font-weight: bold;
        }
        .header {
            color: #1f77b4;
            font-size: 2em;
        }
        .subheader {
            color: #1f77b4;
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
            padding: 0.5em;
            border-radius: 5px;
        }
        .stNumberInput > div > div > input {
            background-color: #ffffff;
            border: 1px solid #1f77b4;
            padding: 0.5em;
            border-radius: 5px;
        }
        .stRadio > div > label, .stCheckbox > div > label {
            color: #2d2d2d;
            font-size: 1em;
        }
        .stButton > button {
            background-color: #1f77b4;
            color: #ffffff;
            border: none;
            padding: 0.75em 1.5em;
            font-size: 1em;
            cursor: pointer;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #a3c6e5;
        }
    </style>
""", unsafe_allow_html=True)

# Main title                 
st.markdown('<h1 class="main-title">🖥️ Client T4FIDS Dashboard</h1>', unsafe_allow_html=True)
about_tab, ins_tab = st.tabs(["About", "Instructions"])
with about_tab:
    st.write('📖 The Client T4FIDS Dashboard is an interactive tool designed for ' \
    'real-time visualization of the training process of a FL-based Intrusion Detection System.')

with ins_tab:
    st.write("""
            📝 First, complete the basic client configuration field by entering a valid server address. 
             You can select the optimizer of your preference through the **Optimizer** tab.
            The completion of the **Advanced** tab is optional. 
             Moreover, the **Custom Model** tab provides the option of designing a custom model on the fly.
            Once these steps are completed, click the "Start Training" button to initiate a connection to the server.
            The FL training will begin when external clients also connect to the server. 
            The dashboard will display the model accuracy in real-time, along with additional evaluation metrics.
            """)

# Reset the acc.json
TMP_SAVE_PATH = 'tmp/streamlit'
ACC_FILE_PATH = Path(TMP_SAVE_PATH) / 'acc.json'
METRICS_FILE_PATH = Path(TMP_SAVE_PATH) / 'metrics.json'
AUX_FILE_PATH = Path(TMP_SAVE_PATH) / 'aux_config.json'
remove_file(ACC_FILE_PATH)

# Sidebar
sidebar = st.sidebar
if "disable_run" not in st.session_state:
    st.session_state.disable_run = False

# Sidebar: Client Configuration
with sidebar:
    #st.image("_static/logo.png", width=250)  # Adjust the width as needed
    st.header('⚙️ :blue[Client Configuration]', divider='blue')

    with st.container(border=True):
        tab_basic, tab_opt, tab_adv = st.tabs(["Basic", "Optimizer", "Advanced"])
        # Input fields
        with tab_basic:
            addr = st.text_input("Server Address:", 
                                 placeholder="Insert server's IP address", 
                                 disabled=st.session_state.disable_run
                                 ) 

            port = st.text_input("Server Port:", 
                                 placeholder="Insert server's port, e.g., 8080",
                                 disabled=st.session_state.disable_run
                                 ) 
            
            path_data = st.text_input("Path to Dataset:", 
                                      placeholder="e.g., data", 
                                      help='Provide the path to the dataset that will be used both for training and testing.', 
                                      disabled=st.session_state.disable_run
                                      )
            lr = st.number_input("Learning Rate:", 
                                 min_value=0.0001, 
                                 value=0.002, step=0.0002, format="%.4f", 
                                 help="Insert value of learning rate", 
                                 disabled=st.session_state.disable_run
                                 )
            
            batch_size = st.number_input("Batch Size:", 
                                 min_value=1, 
                                 value=32, step=2, 
                                 help="Insert value of batch size", 
                                 disabled=st.session_state.disable_run
                                 )
            
            local_epochs = st.number_input("Local Epochs:", 
                                 min_value=1, max_value=20,
                                 value=5, step=1, 
                                 help="Insert value of local epochs", 
                                 disabled=st.session_state.disable_run
                                 )
            
            # Start training button
            start = st.button('Start Training', disabled=st.session_state.disable_run)

            # Alert if the server address is not provided
            if start:
                has_error = False
                if not addr:
                    st.error('Server address is required!', icon="🚨")
                    has_error = True
                if not port:
                    st.error('Server port is required!', icon="🚨")
                    has_error = True
                if not os.path.exists(path_data):
                    st.error('Dataset path does not exist!', icon="🚨")
                    has_error = True
                if not has_error:        
                    st.session_state.disable_run = True

                    # modify config file
                    config['tr_params']['learning_rate'] = lr
                    config['tr_params']['batch_size'] = batch_size
                    config['tr_params']['local_epochs'] = local_epochs
                    config['server']['address'] = addr
                    config['server']['port'] = port
                    config['paths']['data'] = path_data
                    
                    # Save configuration to a file
                    with open(CONFIG_PATH, 'w') as f:
                        json.dump(config, f, indent=4)
                    
                    # Start a new process for running the client script
                    subprocess.Popen(['python', '-m', 'scripts.run_client', 
                                      '--client_type', 'streamlit', 
                                      '--config', 'conf/client/config_dashboard.json']
                    )

            with tab_opt:
                optimizer_name = st.selectbox("Select an Optimizer", 
                                              list(OPTIMIZERS.keys()), 
                                              disabled=st.session_state.disable_run
                                              )
                params = display_optimizer_params(optimizer_name, st.session_state.disable_run)
                # Save configuration to a file
                config['opt'] = {"optimizer": optimizer_name, "params": params}
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(config, f, indent=4)

            with tab_adv:
                if "disabled" not in st.session_state:
                    st.session_state.disabled = True

                features_to_drop = st.empty()
                is_option_features = st.toggle("Remove selected features:", 
                                            help='Whith the removal of specific features, ensure that all clients drop the same feature set.', 
                                            disabled=st.session_state.disable_run
                )
                if is_option_features and os.path.exists(path_data):
                    cols = get_feature_lst(path_data, config['data_kw']['label_keyword'])
                    features_drop = st.multiselect("Select which features to remove:", 
                                                   cols, placeholder='Choose a feature', 
                                                   disabled=st.session_state.disable_run)
                    drop_list = features_drop
                    config['data_kw']['features_to_drop'] = drop_list

                elif is_option_features and not os.path.exists(path_data):
                    st.error("Please, insert a valid dataset path in the corresponding field of the **Basic** tab.", icon="🚨")
                else:
                    features_drop = st.empty()

                is_resampling = st.toggle('Enable upsampling (via SMOTE)', 
                                  help='Upsample training data using SMOTE', 
                                  disabled=st.session_state.disable_run
                                  )
                
                is_override = st.toggle('Enable parameter override', 
                                  help='Enables server to override the client training parameters', 
                                  disabled=st.session_state.disable_run
                                  )
                
                if is_resampling:
                    config['misc']['resample_flag'] = True
                else:
                    config['misc']['resample_flag'] = False

                if is_override:
                    config['misc']['override_params'] = True
                else:
                    config['misc']['override_params'] = False

                with open(CONFIG_PATH, 'w') as f:
                    json.dump(config, f, indent=4)

    st.markdown("""
        <hr>
        <footer style='text-align: center;'>
            &copy; 2025 MetaMind Innovations.
        </footer>
    """, unsafe_allow_html=True)

# Column 2: Real-time Evaluation
st.header('📈 :blue[T4FIDS Real-time Evaluation]', divider='blue')

with st.container(border=True):
    if start and addr and port and os.path.exists(path_data):

        # Waiting spinner
        with st.spinner('Connecting to the server...'):
            wait_for_path(ACC_FILE_PATH)
        st.rerun()
    elif 'acc' not in st.session_state and not st.session_state.disable_run:
        st.info('Evaluation will start upon connection to the server.', icon='ℹ️')
    
    chart_container = st.empty()  # Empty container to hold the chart
    acc = []
    bar = st.empty()
    while True:
        if os.path.exists(ACC_FILE_PATH):
            with open(ACC_FILE_PATH, 'r') as file:
                acc = json.load(file)
            with open(AUX_FILE_PATH, 'r') as f:
                aux_config = json.load(f)
            rounds = list(range(1, len(acc) + 1))
            st.session_state.acc = acc
            if rounds and acc:
                data = {rounds[i]: acc[i] for i in range(len(rounds))}
                data = pd.DataFrame(
                    {
                        "Model Accuracy": acc,
                        "FL Round": rounds,
                        }
                    )
                chart_container.line_chart(data, x='FL Round', y='Model Accuracy')

                # Training progress bar
                bar.progress(rounds[len(rounds)-1]/aux_config["total_rounds"], 
                             text=f'Training in progress ⌛: Round {rounds[len(rounds)-1]}/{aux_config["total_rounds"]}.'
                             )
            else:
                chart_container.empty()  # Clear the container if no data is available
            if len(rounds)==aux_config["total_rounds"]:
                bar.empty()
                break
        elif 'acc' in st.session_state:
            rounds = list(range(1, len(st.session_state.acc) + 1))
            data = {
                "FL Round": rounds,
                "Model Accuracy": st.session_state.acc,
            }
            data_df = pd.DataFrame(data)
            chart_container.line_chart(data_df, x='FL Round', y='Model Accuracy')
            bar.empty()
            break
        else:
            chart_container.empty()  # Clear the container if the metrics file doesn't exist
    
    st.success("✅ FL training has been succesfully completed.")
    metrics_enabled = st.toggle("See evaluation metrics")
    with open(METRICS_FILE_PATH, 'r') as f:
        metrics = json.load(f)
    metrics = tuple(m*100 for m in metrics)
    if metrics_enabled:
        display_metrics(METRICS_FILE_PATH)