# Imports
import argformat
import argparse
import json
import spacy
from pathlib import Path
from spacy.tokens import Doc, Token
from spacy_extensions.embedder.dataset import ContextDataset
from spacy_extensions.embedder.models import *
from spacy_extensions.pipeline import *
from spacy_extensions.vocab import create_vocab_file, file2vocab
from torch.utils.data import DataLoader
import torch
import coreferee
from tqdm import tqdm
from typing import Callable, Union


################################################################################
#                               Operation modes                                #
################################################################################

def vocab(args: argparse.Namespace) -> None:
    """Create vocab file."""
    # Load pipeline
    nlp = spacy.load(args.nlp)

    # Load documents
    docs = list()
    for filename in tqdm(args.files, desc="Loading docs"):
        with open(filename) as infile:
            docs.append(
                Doc(nlp.vocab).from_json(json.load(infile), validate=True)
            )

    # Create dataset
    create_vocab_file(
        outfile = args.output,
        documents = docs,
        include_count = True,
        escape = False,
        topk = args.k,
        frequency = args.freq,
        special = token_base,
        verbose = True,
    )


def dataset(args: argparse.Namespace) -> None:
    """Create dataset from vocab."""
    # Load pipeline
    nlp = spacy.load(args.nlp)
    # Load vocab
    vocab = {
        token: index for index, (token, count) in
        enumerate(file2vocab(args.vocab, escape=True))
    }

    # Load documents
    docs = list()
    for filename in tqdm(args.files, desc="Loading docs"):
        with open(filename) as infile:
            docs.append(
                Doc(nlp.vocab).from_json(json.load(infile), validate=True)
            )

    # Get unknown token
    unk = vocab['[UNK]']

    # Initialise dataset
    data = list()

    # Loop over data
    for doc in tqdm(docs, desc="Creating dataset", total=len(args.files)):
        # Loop over sentences
        for sent in doc.sents:
            # Add sentence
            data.append([vocab.get(token_base(token), unk) for token in sent])
    
    # Write dataset
    with open(args.output, 'w') as outfile:
        for sent in data:
            outfile.write(','.join(map(str, sent)) + '\n')


def train(args: argparse.Namespace) -> None:
    """Train model from dataset."""
    # Load vocab
    vocab = {
        token: index for index, (token, count) in
        enumerate(file2vocab(args.vocab, escape=True))
    }

    # Load data
    data = list()
    with open(args.dataset) as infile:
        for line in infile.read().strip().split('\n'):
            data.append(list(map(int, line.split(','))))

    # Transform to dataset
    context = [5, 5]
    data = ContextDataset(data=data, context=context, pad=vocab['[PAD]'])
    data = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # Select model
    if args.model == 'word2vec-cbow':
        model = Word2vecCBOW(len(vocab))
        loss_function = loss_word2vec_cbow
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == 'word2vec-skipgram':
        model = Word2vecSkipgram(len(vocab))
        loss_function = loss_word2vec_skipgram
        criterion = torch.nn.NLLLoss()
    elif args.model == 'nlm':
        model = NLM(len(vocab), context=sum(context))
        loss_function = loss_word2vec_cbow
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == 'elmo':
        model = ELMo(len(vocab), sum(context))
        loss_function = loss_word2vec_cbow
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Unknown model '{args.model}'")

    # Cast model to device
    model = model.to(args.device)

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    model = model.train()
    for epoch in range(1, args.epochs+1):
        loss_total = 0
        loss_items = 0

        for batch, y_true in (progress := tqdm(data, desc=f"Epoch={epoch}")):
            # Clear gradients
            optimizer.zero_grad()
            # Cast batch to device
            batch = batch.to(args.device)
            y_true = y_true.to(args.device)
            # Compute loss
            loss = loss_function(model, batch, y_true, criterion)
            # Backpropagate
            loss.backward()
            optimizer.step()

            # Update progress
            loss_total += loss.item()
            loss_items += batch.shape[0]
            progress.set_description(
                f"Epoch={epoch} [loss={loss_total/loss_items:.4f}]"
            )

        # Save temporary model
        torch.save(model.to('cpu'), f"{args.output}.{epoch}")
        model = model.to(args.device)

    # Save model
    model = model.to('cpu')
    torch.save(model.state_dict(), args.output)

################################################################################
#                              Embedder training                               #
################################################################################

def loss_word2vec_cbow(
        model: torch.nn.Module,
        X: torch.Tensor,
        y_true: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] 
    ) -> torch.Tensor:
    """Compute loss for Word2vec CBOW model."""
    # Get prediction
    y_pred = model(X)
    # Compute loss and return
    return criterion(y_pred, y_true)


def loss_word2vec_skipgram(
        model: torch.nn.Module,
        X: torch.Tensor,
        y_true: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] 
    ) -> torch.Tensor:
    """Compute loss for Word2vec Skipgram model."""
    # Get prediction
    y_pred = model(y_true)

    # Compute loss
    loss = torch.tensor(0, device=X.device, dtype=torch.float)
    for i in range(y_pred.shape[1]):
        loss += criterion(y_pred[:, i], X[:, i])

    # Return result
    return loss



################################################################################
#                              Auxiliary methods                               #
################################################################################

def read_file(filename: Union[str, Path]) -> str:
    """Read contents from file as text."""
    with open(filename) as infile:
        return infile.read()
        

def token_base(token: Token) -> str:
    """Create token base."""
    return token._.token_base

################################################################################
#                                     Main                                     #
################################################################################

def parse_args():
    """Parse arguments from command line."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description     = 'Train embedder for given documents',
        formatter_class = argformat.StructuredFormatter,
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest = 'mode',
        help = 'mode of operation',
    )

    # Mode vocab
    parser_vocab = subparsers.add_parser('vocab', help='Create vocab')
    parser_vocab.add_argument('nlp', help='pipeline used to create vocab')
    parser_vocab.add_argument('output', help='output path of vocab')
    parser_vocab.add_argument('files', nargs='+',
        help='files to use when creating vocab')
    parser_vocab.add_argument('--k', help='if given, only use k most frequent')
    parser_vocab.add_argument('--freq', type=int, default=1,
        help='minimum frequency')

    # Mode dataset
    parser_dataset = subparsers.add_parser('dataset', help='Create dataset')
    parser_dataset.add_argument('nlp', help='pipeline used to create dataset')
    parser_dataset.add_argument('vocab', help='path to vocab')
    parser_dataset.add_argument('output', help='output path of dataset')
    parser_dataset.add_argument('files', nargs='+',
        help='files to use when creating dataset')

    # Mode train
    parser_train = subparsers.add_parser('train', help='Train model')
    parser_train.add_argument('model', help='model to train', choices=(
        'word2vec-cbow', 'word2vec-skipgram', 'nlm', 'elmo', 'bert',
    ))
    parser_train.add_argument('vocab', help='path to vocab')
    parser_train.add_argument('dataset', help='path to dataset')
    parser_train.add_argument('output', help='output path of trained model')
    # Training parameters
    parser_params = parser_train.add_argument_group('Training parameters')
    parser_params.add_argument('--batch-size', type=int, default=256,
        help='batch size during training')
    parser_params.add_argument('--device', default='auto',
        help='device on which to train')
    parser_params.add_argument('--epochs', type=int, default=100,
        help='number of epochs to train with')
    parser_params.add_argument('--learning-rate', type=float, default=0.01,
        help='learning rate during training')

    # Parse arguments and return
    args = parser.parse_args()

    if 'device' in args and args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Return args
    return args


def main():
    # Parse arguments
    args = parse_args()

    if args.mode == 'vocab':
        vocab(args)
    elif args.mode == 'dataset':
        dataset(args)
    elif args.mode == 'train':
        train(args)


if __name__ == '__main__':
    main()