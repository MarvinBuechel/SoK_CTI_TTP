import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, BertPreTrainedModel, RobertaModel, RobertaConfig

from config import *   # keeps your entity_mapping and MODELS if you still need them elsewhere

class EntityRecognition(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(EntityRecognition, self).__init__()
        self.output_dim = len(entity_mapping)

        # load any transformer (BERT/Roberta/...) by name
        self.bert_layer = AutoModel.from_pretrained(pretrained_model)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # get hidden dim from model's config
        bert_dim = self.bert_layer.config.hidden_size

        # lstm hidden size logic unchanged
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim

        # batch_first kept False so your transposes are still valid
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True
        )
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=self.output_dim)

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample

        # (B, N) → (B, N, E)
        x = self.bert_layer(input_ids=x, attention_mask=attn_masks)[0]

        # (B, N, E) → (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)

        # (N, B, E) → (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x


class MyBert(BertPreTrainedModel):
    def __init__(self, config):
        super(MyBert, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output


class SentenceClassificationBERT(nn.Module):
    def __init__(self, pretrained_model, num_class=2, fine_tune=True):
        super(SentenceClassificationBERT, self).__init__()
        # still using your MyBert wrapper
        self.bert = MyBert.from_pretrained(pretrained_model)

        # Freeze bert layers
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = self.bert.config.hidden_size
        self.classifier = nn.Linear(bert_dim, num_class)

    def forward(self, x, attn_masks):
        outputs = self.bert(x, attention_mask=attn_masks)
        logits = self.classifier(outputs)
        return logits


class MyRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(MyRobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x


class MyRoBerta(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(MyRoBerta, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = MyRobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        output = self.classifier(sequence_output)
        return output


class SentenceClassificationRoBERTa(nn.Module):
    def __init__(self, pretrained_model, num_class=2, fine_tune=True):
        super(SentenceClassificationRoBERTa, self).__init__()
        self.bert = MyRoBerta.from_pretrained(pretrained_model)

        # Freeze bert layers
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = self.bert.config.hidden_size
        self.classifier = nn.Linear(bert_dim, num_class)

    def forward(self, x, attn_masks):
        outputs = self.bert(x, attention_mask=attn_masks)
        logits = self.classifier(outputs)
        return logits
