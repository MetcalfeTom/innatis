from rasa.nlu.training_data import load_data
from rasa_nlu.model import Metadata
from innatis.classifiers.bert_intent_classifier import BertIntentClassifier
import os

model_dir = "home/t_metcalfe/rasa-demo/models/nlu/innatis-full"
model_name = 1555905840
data = "home/t_metcalfe/rasa-demo/train.md"

td = load_data(data)

meta = Metadata.load(model_dir)

bert = BertIntentClassifier.load(model_dir, meta)

bert_model = os.path.join(model_dir, str(model_name))
persisted = os.path.join(model_dir, "finetuned")

bert.continue_training(td, bert_model)
bert.persist(persisted)
