from transformers import (
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification
)

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sentence_transformers import models, SentenceTransformer
import pandas as pd
import numpy as np
from mitreattack.stix20 import MitreAttackData

from const import *


TAU = 0.5


DATASET_LABEL_MAP = {
    ExpDataset.BOSCH_TACTICS: BOSCH_TACTICS_LABELS,
    ExpDataset.BOSCH_SOFTWARE: BOSCH_SOFTWARE_LABELS,
    ExpDataset.BOSCH_GROUPS: BOSCH_GROUP_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_10: BOSCH_TECHNIQUES_10_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_25: BOSCH_TECHNIQUES_25_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_50: BOSCH_TECHNIQUES_50_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_53: BOSCH_TECHNIQUES_53_LABELS,
    ExpDataset.BOSCH_TECHNIQUES: BOSCH_TECHNIQUES_LABELS,
    ExpDataset.TRAM_TECHNIQUES: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_TECHNIQUES_10: TRAM_TECHNIQUES_10_LABELS,
    ExpDataset.TRAM_TECHNIQUES_25: TRAM_TECHNIQUES_25_LABELS,
    ExpDataset.TRAM_TECHNIQUES_SL: TRAM_TECHNIQUES_SL_LABELS,
    ExpDataset.BOSCH_TECHNIQUES_SL: BOSCH_TECHNIQUES_SL_LABELS,
    ExpDataset.BOSCH_ALL: BOSCH_ALL_LABELS,
    ExpDataset.TRAM_AUGMENTED_ARTIFICIAL: TRAM_TECHNIQUES_LABELS,
    ExpDataset.TRAM_AUGMENTED_OOD: TRAM_TECHNIQUES_LABELS,
}


class UnsupportedModel(Exception):
    pass


class MultiLabelTTPSentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels, max_len=MAX_LEN, threshold=TAU):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.labels = labels
        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)

    def __getitem__(self, index):
        sentence = self.data.iloc[index].sentence
        labels = self.data.iloc[index].labels
        encoding = self.tokenizer(
            sentence,
            # return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        labels = labels if len(labels) > 0 else ["NO_TTP"]
        encoded_labels = np.clip(self.lb.transform(labels).sum(axis=0), 0.0, 1.0)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        return item

    def __len__(self):
        return self.len


class SingleLabelTTPSentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels, max_len=MAX_LEN, threshold=TAU):
        self.len = len(dataframe)
        self.data = dataframe.explode("labels").reset_index(drop=True)
        # self.data = self.data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.labels = labels
        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)

    def __getitem__(self, index):
        sentence = self.data.iloc[index].sentence
        labels = self.data.iloc[index].labels
        encoding = self.tokenizer(
            sentence,
            # return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        labels = [labels] if labels in self.labels else ["NO_TTP"]
        encoded_labels = np.clip(self.lb.transform(labels).sum(axis=0), 0, 1)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        return item

    def __len__(self):
        return self.len


class BoschAllDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_ALL_LABELS)


class BoschTechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_LABELS)


class Bosch10TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_10_LABELS)


class Bosch25TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_25_LABELS)


class Bosch50TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_50_LABELS)


class Bosch53TechniquesDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_53_LABELS)


class BoschTacticsDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TACTICS_LABELS)


class BoschGroupsDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_GROUP_LABELS)


class BoschSoftwareDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_SOFTWARE_LABELS)


class TramDataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_LABELS)


class Tram10Dataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_10_LABELS)

class Tram25Dataset(MultiLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_25_LABELS)


class BoschTechniquesDatasetSL(SingleLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, BOSCH_TECHNIQUES_SL_LABELS)


class TramDatasetSL(SingleLabelTTPSentenceDataset):
    def __init__(self, dataframe, tokenizer):
        super().__init__(dataframe, tokenizer, TRAM_TECHNIQUES_SL_LABELS)


def available_models():
    return MODELS + MODEL_SENTENCE_SIM


def model_name_to_folder_name(model_name):
    return model_name.replace("/", "-")


def load_model_for_embedding(model_name):
    if model_name not in available_models():
        raise UnsupportedModel

    print(f"Loading model: {model_name} ...")

    if model_name == "s2w-ai/DarkBERT":
        with open(".darkbert_token", "r") as f:
            token = f.readline().strip()
        transformer = models.Transformer(model_name, model_args={"token": token})
    elif model_name == "priyankaranade/cybert":
        transformer = models.Transformer("local/CyBERT-Base-MLM-v1.1")
    else:
        transformer = models.Transformer(model_name)

    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(), pooling_mode="mean"
    )
    model = SentenceTransformer(modules=[transformer, pooling])
    return model


def load_untrained_model(model_name, dataset_name, multi_label=True):
    dataset = ExpDataset(dataset_name)
    labels = DATASET_LABEL_MAP[dataset]
    num_labels = len(labels)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    if model_name not in available_models() + ["scibert_multi_label_model"]:
        raise UnsupportedModel

    if multi_label == True:
        kwargs = {"problem_type": "multi_label_classification"}
    else:
        kwargs = {"problem_type": "single_label_classification"}

    print(f"Loading model: {model_name} ...")

    if model_name == "s2w-ai/DarkBERT":
        with open(".darkbert_token", "r") as f:
            token = f.readline().strip()
        return RobertaForSequenceClassification.from_pretrained(
            model_name,
            token=token,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        ), AutoTokenizer.from_pretrained(model_name, token=token)
    elif model_name == "priyankaranade/cybert":
        return AutoModelForSequenceClassification.from_pretrained(
            "local/CyBERT-Base-MLM-v1.1",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        ), AutoTokenizer.from_pretrained("local/CyBERT-Base-MLM-v1.1")
    elif model_name == "tram_multi_label_model":
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", max_length=512)
        bert = BertForSequenceClassification.from_pretrained(
            "local/tram_finetuned",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        return bert, tokenizer
    elif model_name == "scibert_multi_label_model":
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", max_length=512)
        bert = BertForSequenceClassification.from_pretrained("local/scibert_multi_label_model")
        return bert, tokenizer
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        ), AutoTokenizer.from_pretrained(model_name)


def load_finetuned_model(model_path):
    return AutoModelForSequenceClassification.from_pretrained(
        model_path
    ), AutoTokenizer.from_pretrained(model_path)


def load_datasets_for_tuning_embedding_threshold(dataset_name):
    dataset_name = ExpDataset(dataset_name)
    if dataset_name == ExpDataset.BOSCH_TECHNIQUES:
        df = pd.read_json("datasets/bosch_train.json")
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES:
        df = pd.read_json("datasets/tram_train.json")
    else:
        print("[!] dataset not found")
        exit()
    print("Dataset: {}".format(df.shape))
    return df


def load_datasets_for_testing_embedding_threshold(dataset_name):
    dataset_name = ExpDataset(dataset_name)
    if dataset_name == ExpDataset.BOSCH_TECHNIQUES:
        df = pd.read_json("datasets/bosch_test.json")
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES:
        df = pd.read_json("datasets/tram_test.json")
    else:
        print("[!] dataset not found")
        exit()
    print("Dataset: {}".format(df.shape))
    return df


def load_datasets_for_finetuning_sentence_model():
    mitre_attack_data = MitreAttackData("datasets/enterprise-attack.json")
    techniques = mitre_attack_data.get_techniques(remove_revoked_deprecated=True)
    print(f"Retrieved {len(techniques)} ATT&CK techniques ...")
    ttps = []
    for t in techniques:
        technique_id = [
            e for e in t.external_references if e.source_name == "mitre-attack"
        ][0].external_id
        content = f"{t.name}:\n{t.description}"
        ttps.append({"id": technique_id, "sentence": content})

    ttps = sorted(ttps, key=lambda x: x["id"])
    df = pd.concat([pd.read_json("datasets/tram_train.json").rename(columns={"doc_title": "document"}), pd.read_json("datasets/bosch_train.json")]).reset_index(drop=True)
    return ttps

def load_augmented(dataset_name, batch_size, tokenizer, test_size=0.2, random_state=0, per_document=False):
    if dataset_name == ExpDataset.TRAM_AUGMENTED_ARTIFICIAL:
        df_train = pd.read_json("datasets/tram_train_augmented_artificial.json")
    elif dataset_name == ExpDataset.TRAM_AUGMENTED_OOD:
        df_train = pd.read_json("datasets/tram_train_augmented_ood.json")


    df_original = pd.read_json("datasets/tram_train.json")
    _, df_val = train_test_split(df_original, test_size=test_size, random_state=random_state)
    df_test = pd.read_json("datasets/tram_test.json")
    train_set = TramDataset(df_train, tokenizer)
    val_set = TramDataset(df_val, tokenizer)
    
    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    val_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
    test_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}

    training_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    if per_document:
        if "doc_title" in df_test.columns:
            grouped = df_test.groupby('doc_title')
        else:
            grouped = df_test.groupby('document')
        df_list = {k: group for k, group in grouped}
        test_sets = {k: TramDataset(d, tokenizer) for k,d in df_list.items()}
        return None, None, {k: DataLoader(d, **test_params) for k,d in test_sets.items()}
    else:
        test_set = TramDataset(df_test, tokenizer)
        testing_loader = DataLoader(test_set, **test_params)
    
    return training_loader, val_loader, testing_loader


def load_datasets(dataset_name, batch_size, tokenizer, test_size=0.2, random_state=0, per_document=False):
    dataset_name = ExpDataset(dataset_name)
    dataset_class = None
    df_train_name = None
    df_test_name = None
    if dataset_name == ExpDataset.BOSCH_TECHNIQUES:
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschTechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TACTICS: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschTacticsDataset
    elif dataset_name == ExpDataset.BOSCH_GROUPS: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschGroupsDataset
    elif dataset_name == ExpDataset.BOSCH_SOFTWARE: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschSoftwareDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_10: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch10TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_25: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch25TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_50: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch50TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_53: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = Bosch53TechniquesDataset
    elif dataset_name == ExpDataset.BOSCH_TECHNIQUES_SL: 
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschTechniquesDatasetSL
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = TramDataset
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES_10:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = Tram10Dataset
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES_25:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = Tram25Dataset
    elif dataset_name == ExpDataset.TRAM_TECHNIQUES_SL:
        df_train_name = "datasets/tram_train.json"
        df_test_name = "datasets/tram_test.json"
        dataset_class = TramDatasetSL
    elif dataset_name == ExpDataset.BOSCH_ALL:
        df_train_name = "datasets/bosch_train.json"
        df_test_name = "datasets/bosch_test.json"
        dataset_class = BoschAllDataset
    elif dataset_name in [ExpDataset.TRAM_AUGMENTED_ARTIFICIAL, ExpDataset.TRAM_AUGMENTED_OOD]:
        return load_augmented(dataset_name, batch_size, tokenizer, test_size, random_state, per_document)
    
    df = pd.read_json(df_train_name)
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
    df_test = pd.read_json(df_test_name)

    train_set = dataset_class(df_train, tokenizer)
    val_set = dataset_class(df_val, tokenizer)

    train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    val_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}
    test_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}

    training_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    if per_document:
        if "doc_title" in df_test.columns:
            grouped = df_test.groupby('doc_title')
        else:
            grouped = df_test.groupby('document')
        df_list = {k: group for k, group in grouped}
        test_sets = {k: dataset_class(d, tokenizer) for k,d in df_list.items()}
        return None, None, {k: DataLoader(d, **test_params) for k,d in test_sets.items()}
    else:
        test_set = dataset_class(df_test, tokenizer)
        testing_loader = DataLoader(test_set, **test_params)

    return training_loader, val_loader, testing_loader
