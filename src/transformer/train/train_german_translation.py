import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import torch.functional as F
from tqdm import tqdm

from transformer.models.encoder_decoder_transformer import EncoderDecoderTransformer
from transformer.data.german_translation import get_batches

def get_de_en_dataset():
    dataset = load_dataset("bentrevett/multi30k")

    all_en = " ".join([pair['en'] for pair in dataset['train']])
    all_de = " ".join([pair['de'] for pair in dataset['train']])

    en_chars = sorted(list(set(all_en))) + ["<PAD>", "<BOS>", "<EOS>"]
    de_chars = sorted(list(set(all_de))) + ["<PAD>"]

    x_vocab_size = len(en_chars)
    y_vocab_size = len(de_chars)

    en_stoi = {c:i for i, c in enumerate(en_chars)}
    en_itos = {i:c for i, c in enumerate(en_chars)}

    de_stoi = {c:i for i, c in enumerate(de_chars)}
    de_itos = {i:c for i, c in enumerate(de_chars)}

    en_encode = lambda s: [en_stoi[c] for c in s]
    de_encode = lambda s: [de_stoi[c] for c in s]

    en_decode = lambda l: "".join([en_itos[i] for i in l])
    de_decode = lambda l: "".join([de_itos[i] for i in l])

    # First encode each training example
    sources = [torch.tensor(de_encode(pair['de']), dtype=torch.long) for pair in dataset['train']]

    translations = [torch.tensor(
        [en_stoi["<BOS>"]] + en_encode(pair['en']) + [en_stoi["<EOS>"]], dtype=torch.long
        ) for pair in dataset['train']]

    return {
        'translations': translations,
        'sources': sources,
        'en_stoi': en_stoi,
        'de_stoi': de_stoi
    }

def main():

    data = get_de_en_dataset()

    translations = data['translations']
    sources = data['sources']
    en_stoi = data['en_stoi']
    de_stoi = data['de_stoi']


    # model hyperparameters
    x_embedding_dim = 128
    y_embedding_dim = 128

    x_block_size = max([len(i) for i in translations])
    y_block_size = max([len(i) for i in sources])

    x_vocab_size =  x_vocab_size
    y_vocab_size = y_vocab_size

    num_enc_sa_heads = 4
    num_enc_blocks = 3

    num_dec_sa_heads = 4
    num_dec_ca_heads = 4
    dec_ca_head_dim = 32
    dec_ca_value_dim = 32
    num_dec_blocks = 3
    dropout = 0.2

    # training hyperparameters
    torch.manual_seed(42)
    batch_size = 32
    train_iters = 100
    print_iters = train_iters // 10
    lr=3e-4

    encdec = EncoderDecoderTransformer(
        x_embedding_dim,
        y_embedding_dim,
        x_block_size,
        y_block_size,
        x_vocab_size,
        y_vocab_size,
        num_enc_sa_heads,
        num_enc_blocks,
        num_dec_blocks,
        num_dec_sa_heads,
        num_dec_ca_heads,
        dec_ca_head_dim,
        dec_ca_value_dim,
        dropout)

    optim = torch.optim.AdamW(encdec.parameters(), lr=lr)

    x_pad_id = en_stoi["<PAD>"]
    y_pad_id = de_stoi["<PAD>"]


    for _ in tqdm(range(train_iters)):

        x, y, target = get_batches(batch_size)

        logits = encdec(x, y, x_pad_id, y_pad_id)

        loss = F.cross_entropy(logits.reshape(-1, x_vocab_size), target.reshape(-1),ignore_index=en_stoi["<PAD>"])

        optim.zero_grad()
        loss.backward()
        optim.step()

        if _ % print_iters == 0:
            print(loss.item())


    torch.save(
    {
        "model": encdec.state_dict(),
        "optim": optim.state_dict(),
        "epoch": train_iters,
        "loss": loss.item()
        },
        "artifacts/de_to_en_checkpoint.pth"
    )
