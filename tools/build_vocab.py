import torch


vocab_input_file = "./data/vocab.txt"
vocab_output_file = "./data/ecdt2019.vocab.pt"

itos = ['<pad>', '<unk>', '<bos>', '<eos>']
with open(vocab_input_file, 'r', encoding='utf-8') as fr:
    for line in fr:
        itos.append(line.strip())
print("itos:", len(itos))
vocab_dict = {"itos": itos,
              "embeddings": None
              }
vocab = {"src": vocab_dict,
         "tgt": vocab_dict,
         "cue": vocab_dict
         }

torch.save(vocab, vocab_output_file)