import torch
import numpy as np

import time

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random
import os
import sys
from transformers import get_scheduler

import loader
from train_common import *
from torch import nn


def start_training(model, training_loader, val_loader, output_dir, model_params):
    """
    Actual training routine.
    """
    learning_rate = model_params["learning_rate"]
    freeze_layers = model_params["freeze_layers"]
    pos_weight = model_params["pos_weight"]
    end_factor = model_params.get("end_factor", 0.7)
    model_file = os.path.join(output_dir, "model_chkp")
    epochs = model_params["epochs"]

    if int(freeze_layers) != 0:
        breakpoint()
        inner_mdl = getattr(model, "bert", None) or getattr(model, "roberta", None)
        for param in inner_mdl.embeddings.parameters():
            param.requires_grad = False
        for param in inner_mdl.encoder.layer[:freeze_layers].parameters():
            param.requires_grad = False

    # Define the optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # Total training steps
    total_steps = len(training_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_steps)
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings
    training_stats = []

    # Measure the total training time for the whole run
    total_t0 = time.time()
    step_no_improv = 0
    best_eval_loss = 10000

    g = open(os.path.join(output_dir, "classification_reports.txt"), "w")
    if pos_weight != 0:
        pos_weight = torch.full((model.num_labels,), pos_weight).to(device)
    else:
        pos_weight = None
    
    # For each epoch...
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set

        print("\n>======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        # Measure how long the training epoch takes
        t0 = time.time()

        # Reset for this epoch
        tr_loss, tr_accuracy, tr_f1 = 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        # Put the model into training mode
        model.train()

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # For each batch of training data...
        for batch in tqdm(training_loader, desc="Training"):
            # Unpack this training batch from the dataloader
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.float64)

            # backward pass
            optimizer.zero_grad()

            # Perform a forward pass
            result = model(
                input_ids=ids, attention_mask=mask, labels=labels
            )  # , return_dict=True)

            # breakpoint()
            # loss = result.loss
            loss = criterion(result.logits, labels)
            # loss = criterion(torch.where(result.logits.sigmoid() > TAU, 1.0, 0.0), l)
            tr_loss += loss.item()
            tr_logits = result.logits

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            # compute training accuracy
            # active_logits = tr_logits.view(-1, model.num_labels)

            probs = torch.nn.functional.sigmoid(tr_logits)
            predictions = torch.where(probs > TAU, 1.0, 0.0)

            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(
                labels.cpu().numpy(), predictions.cpu().numpy()
            )
            tmp_tr_f1 = f1_score(
                labels.cpu().numpy(),
                predictions.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )

            tr_accuracy += tmp_tr_accuracy
            tr_f1 += tmp_tr_f1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            # print(f"\tCurrent Learning Rate: {current_lr:.8f}")

        # Calculate the average loss over all of the batches
        avg_train_loss = tr_loss / len(training_loader)

        # Calculate the average accuracy over all of the batches
        tr_accuracy = tr_accuracy /  len(training_loader)
        tr_f1 = tr_f1 /  len(training_loader)

        # Measure how long this epoch took
        training_time = format_time(time.time() - t0)

        print("\n\tAverage training loss: {0:.8f}".format(avg_train_loss))
        print("\tAverage training accuracy: {0:.8f}".format(tr_accuracy))
        print("\tAverage training weighted f1-score: {0:.8f}".format(tr_f1))
        print("\tTraining epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure the performance on
        # the validation set
        print("\nRunning Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
        nb_eval_examples, nb_eval_steps = 0, 0
        eval_preds, eval_labels = [], []

        # Evaluate data for one epoch
        for batch in tqdm(val_loader, desc="Validation"):

            # Unpack this training batch from dataloader
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.float64)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training)
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model(
                    input_ids=ids, attention_mask=mask, labels=labels, return_dict=True
                )

            # breakpoint()
            # loss = result.loss
            loss = criterion(result.logits, labels)
            eval_logits = result.logits

            # Accumulate the validation loss
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            # active_logits = eval_logits.view(-1, model.num_labels)

            probs = torch.nn.functional.sigmoid(eval_logits)
            predictions = torch.where(probs > TAU, 1.0, 0.0)

            eval_labels.extend(labels.cpu().numpy())
            eval_preds.extend(predictions.cpu().numpy())

            tmp_eval_accuracy = accuracy_score(
                labels.cpu().numpy(), predictions.cpu().numpy()
            )
            tmp_eval_f1 = f1_score(
                labels.cpu().numpy(),
                predictions.cpu().numpy(),
                average="weighted",
                zero_division=0,
            )

            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1

        cr = classification_report(eval_labels, eval_preds, zero_division=0)
        g.write(f"> epoch: {epoch_i}\n")
        g.write(cr)
        g.write("\n")
        # Report the final accuracy for this validation run
        avg_val_accuracy = eval_accuracy / len(val_loader)
        avg_val_f1 = eval_f1 / len(val_loader)
        print("\tAccuracy: {0:.8f}".format(avg_val_accuracy))
        print("\tWeighted f1: {0:.8f}".format(avg_val_f1))

        # Calculate the average loss over all of the batches
        avg_val_loss = eval_loss / len(val_loader)

        # Measure how long the validation run took
        validation_time = format_time(time.time() - t0)

        print("\tValidation Loss: {0:.8f}".format(avg_val_loss))
        print("\tValidation took: {:}".format(validation_time))

        # Save best model
        if avg_val_loss < best_eval_loss:
            step_no_improv = 0
            print(f"\t[!] Saving model at {model_file}")
            torch.save(model, model_file)
            best_eval_loss = avg_val_loss
        else:
            step_no_improv += 1

        # Record all statistics from this epoch
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Valid. F1": avg_val_f1,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

        if step_no_improv >= PATIENCE:
            print("No improvements for %s epochs" % PATIENCE)
            break

    g.close()
    print("\n[!] Training complete")
    print(f"Training took {format_time(time.time() - total_t0)} (h:mm:ss)")
    make_report(output_dir, model_params, training_stats)


if __name__ == "__main__":
    config_file = sys.argv[1]
    device = sys.argv[2]
    index_start = int(sys.argv[3])
    n_to_read = int(sys.argv[4])

    if len(sys.argv) < 1:
        print("usage: python train_multilabel.py CONFIG_FILE")

    if device == "cpu":
        print("[!] Running on CPU!")

    exp_configs = parse_config(config_file)
    exp_configs = exp_configs[index_start:index_start+n_to_read]

    for conf_id, model_name, dataset_name, model_params in exp_configs:
        try:
            print("TRAIN MULTILABEL:")
            print(
                f"conf_id: {conf_id}, model_name: {model_name}, dataset_name: {dataset_name}"
            )
            print(f"model_params:\n{model_params}")
            mdl, tokenizer = loader.load_untrained_model(model_name, dataset_name)
            mdl.to(device)
            batch_size = model_params["batch_size"]
            sanitized_name = loader.model_name_to_folder_name(model_name)
            folder_name = "%s_%s_%s" % (conf_id, dataset_name, sanitized_name)
            output_dir = os.path.join("fine_tuned", "multi_label", folder_name)
            model_file = os.path.join(output_dir, "model_chkp")
            os.makedirs(output_dir, exist_ok=True)
            # # display_df_stats(training_stats)
            train_loader, val_loader, _ = loader.load_datasets(
                dataset_name, batch_size, tokenizer, test_size=0.2
            )
            start_training(mdl, train_loader, val_loader, output_dir, model_params)
            best_model = torch.load(model_file)
            model_to_save = (
                best_model.module if hasattr(best_model, "module") else best_model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            os.remove(model_file)
        except Exception as e:
            raise e
            print("[!] ERROR! Skipping")
