# Transformer

PyTorch implementation of the Transformer architecture.

Implements multi-head attention by computing all heads in parallel for both self-attention and cross-attention.

Demonstrates how to combine these layers to build decoder-only, encoder-only, and encoder-decoder Transformers.

# Quick Start

Install [uv](https://docs.astral.sh/uv/) and run either command from the repository root:

```bash
uv run train_german_translation
uv run train_decoder
```

The translation script downloads Multi30K on first use. The Shakespeare script uses the included Tiny Shakespeare dataset. Both scripts save checkpoints to `artifacts/`.

# Demos

## German-to-English Machine Translation

We use an encoder-decoder Transformer to train a German-to-English translation model with character-level tokenization on the [Multi30K dataset](https://huggingface.co/datasets/bentrevett/multi30k). The examples below are from the held-out test set and use a model trained for approximately 15 minutes on a Google Colab T4 GPU.

```
German: Drei Leute sitzen in einer Höhle.
Target: Three people sit in a cave.
Prediction: Three people sitting on a side in a hold.
```

```
German: Ein Mädchen in einem Jeanskleid läuft über einen erhöhten Schwebebalken.
Target: A girl in a jean dress is walking along a raised balance beam.
Prediction: A girl in a black and blue shirt and shirt on a bushe building.
```

```
German: Ein Kind planscht im Wasser.
Target: A child is splashing in the water
Prediction: A child is playing a water a water.
```

## Shakespeare Next-Character Prediction

We use a decoder-only Transformer to train a character-level language model on Shakespeare's works. The model was trained for approximately 30 minutes on a Google Colab T4 GPU.

Sample generated text:

```
MARCIUS:
I have remember
I am: rich, which is't not here so fraild a varted
That I were wotter on agree
To speak, good with intents.

CORIOLANUS:
Such you plainless trusty dischery
More one infortunes?

POMPEY:
No, by; it will so his great and revely;
But nothing Rome, by poor complainly in soul,
To know it indeed linew to execution,
And he not, if my treasure is death in my firmile
To wheppish I live, my Lucious, whose is my soul,
To the tribune o'erpation.

DUKE VINCENTIO:
What's thou midwar
```

# References

## Papers

[Attention Is All You Need (Vaswani et al., NeurIPS 2017)](https://arxiv.org/abs/1706.03762)

[On Layer Normalization in the Transformer Architecture (Xiong et al., ICML 2020)](https://arxiv.org/abs/2002.04745)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)

## Videos

[Transformers, the tech behind LLMs | Deep Learning Chapter 5 (3Blue1Brown)](https://www.youtube.com/watch?v=wjZofJX0v4M&vl=en)

[Let's build GPT: from scratch, in code, spelled out. (Andrej Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Repositories

[ng-video-lecture GitHub Repository (Andrej Karpathy)](https://github.com/karpathy/ng-video-lecture)