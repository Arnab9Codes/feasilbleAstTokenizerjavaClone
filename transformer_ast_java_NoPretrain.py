import torch
from torch.utils.data.dataset import Dataset
import transformers
import tokenizers
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Split
import json
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names,get_dataset_config_names
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import BertConfig, BertLMHeadModel
from transformers import Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer

# loading codesearch net java dataset
dataset_codeSearch_java=load_dataset("code_search_net","java")
djava=dataset_codeSearch_java
print("loaded the dataset.")
unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]


def prepare_tokenizer_trainer(alg='byteBPE'):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg=='byteBPE':
        tokenizer = ByteLevelBPETokenizer()
    
    #tokenizer.pre_tokenizer = Split(';','removed')
    return tokenizer

tokenizer=prepare_tokenizer_trainer('byteBPE')
tokenizer.train(files='./tokens1to25k_java_ast.txt',vocab_size=50000, min_frequency=2,
                show_progress=True,
                special_tokens=[
                                "<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>"])

tokenizer.save('./ASTbpeTokens1to25k_java')

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

fast_tokenizer.mask_token='<mask>'
fast_tokenizer.pad_token='<pad>'

class CustomDataset(Dataset):
    def __init__(self, evaluate: bool = False):

        self.examples = []
        self.sample_size=2000 # change it to something meaningful later

        self.src_files = djava
        self.evaluate=evaluate
        
        if self.evaluate:
            for i in range(self.sample_size+1, (self.sample_size+201),1):#2001,201
                sentences=djava['train']['whole_func_string'][i].split('\n')
                self.examples+=[t.ids for t in tokenizer.encode_batch(sentences)]
                print("self", self.evaluate+1)
        else:
            for i in range(0,self.sample_size,1):
                sentences=djava['train']['whole_func_string'][i].split('\n')
                self.examples+=[t.ids for t in tokenizer.encode_batch(sentences)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i], dtype=torch.int64)

d=CustomDataset(evaluate=False)
print("loading training dataset is done.")
e=CustomDataset(evaluate=True)
print("loading eval dataset is done.", e.__len__())

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=fast_tokenizer, mlm=True, mlm_probability=0.15, return_tensors='pt'
)

config = BertConfig(50000, hidden_size=30,
                    num_hidden_layers=2, num_attention_heads=2, is_decoder=True,
                    add_cross_attention=True)
model = BertLMHeadModel(config)


training_args = TrainingArguments(
    output_dir="./ast_transformer_no_pretrain",
    overwrite_output_dir=True,
    num_train_epochs=100,#100,
    do_train=True,
    per_gpu_train_batch_size=128,
    save_steps=500,
    save_total_limit=50,
    logging_steps =100,
    eval_steps=500,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=d,
    eval_dataset=e,
    data_collator=data_collator
)

trainer.train()
print(trainer.state.log_history)
