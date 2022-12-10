import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModel

from embedding_utils import bert_embedding
from clustering_utils import plot_hierarchical_image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load data
data_path = "./data/detail.json"
data = json.loads(Path(data_path).read_text())
answers = [example["answer"] for example in data if example["category"] != "None"]
questions = [example["question"] for example in data if example["category"] != "None"]
categories = [example["category"] for example in data if example["category"] != "None"]

# Initialize bert tokenizer and model
model_name_or_path = "bert-base-chinese"
cache_dir = "/nfs/nas-7.1/chchen/cache/huggingface/"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

#############################################
# Clustering by questions
#############################################

save_path = "./outputs/clustering_by_question.png"
X = bert_embedding(model, tokenizer, x=questions, y=categories)
plot_hierarchical_image(X, Y=sorted(list(set(categories))), save_path=save_path)

#############################################
# Clustering by answers
#############################################

save_path = "./outputs/clustering_by_answer.png"
X = bert_embedding(model, tokenizer, x=answers, y=categories)
plot_hierarchical_image(X, Y=sorted(list(set(categories))), save_path=save_path)

#############################################
# Clustering by surface form
#############################################

save_path = "./outputs/clustering_by_surface.png"
surface = sorted(list(set(categories)))
X = bert_embedding(model, tokenizer, x=surface, y=surface)
plot_hierarchical_image(X, Y=surface, save_path=save_path)