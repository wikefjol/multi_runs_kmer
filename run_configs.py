import os
import torch
import json
import pprint

from sklearn.model_selection import train_test_split
from transformers import BertModel, BertConfig
from torch.utils.data import DataLoader
from datetime import datetime

# Data handeling
from src.data_module.dataset import MLMDataset, ClassificationDataset
from src.data_module.data_tools import filter_taxonomy, fasta2pandas
from src.utils.vocab import Vocabulary, KmerVocabConstructor

# Factory
from src.factories.preprocessing_factory import create_preprocessor

# Model things
from src.model.backbone import Bertax, ModularBertax
from src.model.encoders import LabelEncoder
from src.model.heads import MLMHead, SingleClassHead

# Training things
from src.train.trainers import MLMtrainer, ClassificationTrainer

# Logging things
from src.utils.logging_utils import setup_logging, plot_training_performance, plot_finetuning_comparison, plot_pretraining_comparison

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def run_config(file_path, run_dir, run_name):

    ########################################################################################################################
    ## Load config #########################################################################################################
    ########################################################################################################################

    CONFIG = load_config(file_path)


    log_dir = os.path.join(run_dir, "logs")
    
    # Set up system logger
    system_logger = setup_logging(
        log_dir = log_dir,
        system_level = CONFIG["system_log_lvl"]
        )
    
    # Set up jsonsls for training metrics
    metrics_file_pre = os.path.join(log_dir, "pretraining.jsonl")
    metrics_file_fine = os.path.join(log_dir, "finetuning.jsonl")
    
    ########################################################################################################################
    ## Extract key values from config, to use in entire notebook ###########################################################
    ########################################################################################################################

    k = CONFIG["preprocessing"]["tokenization"]["k"]
    alphabet = CONFIG["alphabet"]
    optimal_length = CONFIG["max_sequence_length"] // k


    test_size = CONFIG["test_size"]
    target_column = CONFIG["target_labels"][0]

    if len(CONFIG["target_labels"])!=1:
        system_logger.error("Too many target labels. Only expected one, got: ", CONFIG["target_labels"])
    else:
        target_label = CONFIG["target_labels"][0]

    masking_percentage = CONFIG["masking_percentage"]
    pre_training_batch_size = CONFIG["batch_size"]["pre_training"]
    fine_tuning_batch_size = CONFIG["batch_size"]["fine_tuning"]
    modification_probability = CONFIG["preprocessing"]["augmentation"]["training"]["modification_probability"]

    ########################################################################################################################
    ## Expand some parts of  the config given the key values ###############################################################
    ## to avoid that we need to have the same values twice in the config, but still access them where needed in factory ####
    ########################################################################################################################

    CONFIG["model"]["max_position_embeddings"] = optimal_length + 2
    CONFIG["preprocessing"]["padding"]["optimal_length"] = optimal_length
    CONFIG["preprocessing"]["truncation"]["optimal_length"] = optimal_length

    CONFIG["preprocessing"]["tokenization"]["alphabet"] = alphabet
    CONFIG["preprocessing"]["augmentation"]["training"]["alphabet"] = alphabet
    CONFIG["preprocessing"]["augmentation"]["evaluation"]["alphabet"] = alphabet
        


    
    # Log the "Run started" message once
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_logger.info(f"Run started: {timestamp}")

    ########################################################################################################################
    ## Construct vocabulary ################################################################################################
    ########################################################################################################################

    #LATER: Write a function that creates the vocabulary given the config
    constructor = KmerVocabConstructor(
        k = k,
        alphabet = alphabet
        )

    vocab = Vocabulary()
    vocab.build_from_constructor(constructor, data=[])
     # Log the "Run started" message once

    ########################################################################################################################
    ## Create preprocessors ################################################################################################
    ########################################################################################################################

    train_preprocessor = create_preprocessor(
        config = CONFIG["preprocessing"],
        vocab = vocab,
        training = True
        )

    eval_preprocessor = create_preprocessor(
        config = CONFIG["preprocessing"],
        vocab = vocab,
        training = False
        )
    
    ##################################################################
    ## Data preparation ##############################################
    ##################################################################
    
    ## Get all data
    all_data = fasta2pandas(CONFIG["FILE_PATH"])
    if CONFIG["small_set"]:
        # maybe use subset, just for testing purposes
        all_data = all_data[:CONFIG["n_test"]]

    ## Filter data for finetuning
    filtered_data = filter_taxonomy(
        df = all_data,
        startAt = target_column,
        endAt = target_column,
        phylumCertainty = True
        )

    ##  Identify how many classes exist in the finetuning data set
    num_classes = len(list(set(filtered_data[target_label])))

    ## Naive pretraining datasplit based on unsplit data: 
    pretrain_sequences, preval_sequences = train_test_split(
        all_data["sequence"],
        test_size = test_size,
        random_state = 42
    )

    ## Finetune datasplit based on filtered data:
    # LATER: Split data better
    finetrain_data, fineval_data = train_test_split(
        filtered_data,
        test_size = test_size,
        random_state = 69
        )

    system_logger.info(f" -- Number of pre-training sequences: {len(pretrain_sequences)}")
    system_logger.info(f" -- Number of validation sequences: {len(preval_sequences)}")

    ##################################################################
    ## Setup Datasets ################################################
    ##################################################################
    #TODO: I cant wrap these, since they are not functions. Help me create logging calls that behave similar to the with_logging wrapper here
    ## Pretraining datasets
    pretrain_dataset = MLMDataset(
        df = pretrain_sequences,
        preprocessor = train_preprocessor,
        masking_percentage = masking_percentage
    )

    preval_dataset = MLMDataset(
        df = preval_sequences,
        preprocessor = eval_preprocessor,
        masking_percentage = masking_percentage
    )

    ## Finetuning datasets
    label_encoder = LabelEncoder(filtered_data[target_label])

    finetrain_dataset = ClassificationDataset(
        df = finetrain_data,
        preprocessor = train_preprocessor,
        label_encoder = label_encoder,
        target_column = target_column
        )

    fineval_dataset = ClassificationDataset(
        df = fineval_data,
        preprocessor = eval_preprocessor,
        label_encoder = label_encoder,
        target_column = target_column
        )

    ##################################################################
    ## Setup Dataloaders #############################################
    ##################################################################
    pretrain_loader = DataLoader(
        dataset = pretrain_dataset,
        batch_size = pre_training_batch_size,
        shuffle = True
    )

    preval_loader = DataLoader(
        dataset = preval_dataset,
        batch_size = pre_training_batch_size,
        shuffle = True
    )

    finetrain_loader = DataLoader(
        dataset=finetrain_dataset,
        batch_size=fine_tuning_batch_size,
        shuffle=True
        )

    fineval_loader = DataLoader(
        dataset=fineval_dataset,
        batch_size=fine_tuning_batch_size,
        shuffle=True
        )

    ########################################################################################################################
    ## Create encoder ######################################################################################################
    ########################################################################################################################
    encoder_config = BertConfig(
        vocab_size = len(vocab),
        hidden_size = CONFIG["model"]["hidden_size"],
        num_hidden_layers = CONFIG["model"]["num_layers"],
        num_attention_heads = CONFIG["model"]["num_attention_heads"],
        intermediate_size = CONFIG["model"]["intermediate_size"],
        max_position_embeddings = CONFIG["model"]["max_position_embeddings"],
        hidden_dropout_prob = CONFIG["model"]["dropout_rate"],
        attention_probs_dropout_prob = CONFIG["model"]["dropout_rate"]
    )

    encoder = BertModel(encoder_config)

    ########################################################################################################################
    ## Create heads ########################################################################################################
    ########################################################################################################################
    mlm_head = MLMHead(
        in_features = CONFIG["model"]["hidden_size"],
        hidden_layer_size = CONFIG["model"]["hidden_size"] // 2,
        out_features = len(vocab),
        dropout_rate = CONFIG["model"]["mlm_dropout_rate"]
    )

    classification_head = SingleClassHead(
        in_features = CONFIG["model"]["hidden_size"],
        hidden_layer_size = CONFIG["model"]["hidden_size"] // 2,
        out_features = num_classes,
        dropout_rate = CONFIG["model"]["dropout_rate"]
    )

    ########################################################################################################################
    ## Create model ########################################################################################################
    ########################################################################################################################
    system_logger.info(
    "Assembling model with the following components:\n"
    f"Encoder:\n{pprint.pformat(encoder, indent=2)}\n"
    f"MLM Head:\n{pprint.pformat(mlm_head, indent=2)}\n"
    f"Classification Head:\n{pprint.pformat(classification_head, indent=2)}"
    )

    model = ModularBertax(
        encoder = encoder,
        mlm_head = mlm_head,
        classification_head = classification_head
    )

    ########################################################################################################################
    ## Create trainers #####################################################################################################
    #######################################################################################################################
    pre_trainer = MLMtrainer(
        model=model,
        train_loader=pretrain_loader,
        val_loader=preval_loader,
        num_epochs=CONFIG["max_epochs"]["pre_training"],
        patience=CONFIG["patience"]["pre_training"],
        best_weight_save_path = run_dir + "/best_pre_training_weights.pt",
        metrics_jsonl_path=metrics_file_pre
    )

    fine_tuner = ClassificationTrainer(
        model=model,
        train_loader=finetrain_loader,
        val_loader=fineval_loader,
        num_epochs=CONFIG["max_epochs"]["fine_tuning"],
        patience=CONFIG["patience"]["fine_tuning"],
        best_weight_save_path = run_dir + "/best_fine_tuning_weights.pt",
        metrics_jsonl_path=metrics_file_fine
    )
    ########################################################################################################################
    ## Pretrain model ######################################################################################################
    ########################################################################################################################
    # Log all relevant information for pretraining
    system_logger.info(
        f"Starting pretraining with the following settings: "
        f"num_epochs={CONFIG['max_epochs']['pre_training']}, "
        f"batch_size={CONFIG['batch_size']['pre_training']}, "
        f"masking_percentage={CONFIG['masking_percentage']}"
    )

    # Train the model
    pre_trainer.train()

    #Plot pretraining performance
    plot_training_performance(
        metrics_file_name="pretraining.jsonl",
        run_name=run_name,
    )

    ########################################################################################################################
    ## Finetune model ######################################################################################################
    ########################################################################################################################
    # Log all relevant information for fine-tuning
    system_logger.info(
        f"Starting fine-tuning with the following settings: "
        f"num_epochs={CONFIG['max_epochs']['fine_tuning']}, "
        f"batch_size={CONFIG['batch_size']['fine_tuning']}, "
        f"target_labels={CONFIG['target_labels']}"
    )

    # Switch to classification mode
    model.classifyMode()

    # Train the model
    fine_tuner.train()

    # Plot finetuning performance
    plot_training_performance(
        metrics_file_name="finetuning.jsonl",
        run_name=run_name,
    )

    ########################################################################################################################
    ## Save the expanded config ###########################################################################################
    ########################################################################################################################
    expanded_config_path = os.path.join(run_dir, "expanded_config.json")
    with open(expanded_config_path, "w") as config_file:
        json.dump(CONFIG, config_file, indent=4)

    log_message = f"Saved expanded config to {expanded_config_path}"
    system_logger.info(log_message)
    print(log_message)


if __name__ == "__main__":
    CONFIG_DIR = "configs"
    RUNS_DIR = "runs"
    
    # Ensure the directory exists
    if not os.path.exists(CONFIG_DIR):
        print(f"Config directory '{CONFIG_DIR}' not found.")
        exit(1)

    # Ensure the runs directory exists
    os.makedirs(RUNS_DIR, exist_ok=True)

    # Iterate through all JSON files in the directory
    for config_file in os.listdir(CONFIG_DIR):
        if config_file.endswith(".json"):
            
            
            config_path = os.path.join(CONFIG_DIR, config_file)
            run_name = os.path.splitext(config_file)[0]  # Folder name = config file name (without extension)
            run_dir = os.path.join(RUNS_DIR, run_name)

        
            # Create a folder for the current run
            os.makedirs(run_dir, exist_ok=True)
            
            print(f"Running config: {config_path}")
            try:
                run_config(config_path, run_dir, run_name)
                plot_finetuning_comparison(runs_dir = "runs")
                plot_pretraining_comparison(runs_dir = "runs")
            except Exception as e:
                print(f"Error processing config '{config_file}': {e}")