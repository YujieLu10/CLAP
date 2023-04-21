import spacy
import nltk
from rouge import Rouge
from datasets import load_metric
from typing import List
from sentence_transformers import util as st_utils
from moverscore_v2 import word_mover_score
from collections import defaultdict, Counter
import numpy as np
nlp_encoder = spacy.load('en_core_web_md')

def calc_text_distance(query_str, value_str, translation_lm, device):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    value_embedding = translation_lm.encode(value_str, convert_to_tensor=True, device=device)
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, value_embedding)[0].detach().cpu().numpy()
    return cos_scores

def calc_textemb_distance(query_emb, value_emb):
    cos_scores = st_utils.pytorch_cos_sim(query_emb, value_emb)[0].detach().cpu().numpy()
    return float(cos_scores)

def get_metric_result(task, predicted_intent, task_eval_groundtruth, task_eval_predict, translation_lm, device):
    task_eval_groundtruth = task_eval_groundtruth.replace('.', ' ')
    task_eval_predict = task_eval_predict.replace('.', ' ')
    sentence_bleu = nltk.translate.bleu_score.sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split())
    sim = nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict))
    rouge = Rouge()
    scores = rouge.get_scores(task_eval_predict, task_eval_groundtruth)
    rouge_l_f = scores[0]["rouge-l"]["f"]
    bertscore = load_metric("bertscore")
    try:
        bert_results = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased")
        bert_score_norm = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased", lang="en", verbose=False, rescale_with_baseline=True)["f1"][0]
    except:
        bert_score_norm = 0
    intent_score = calc_text_distance(task, predicted_intent, translation_lm, device)
    return sentence_bleu, sim, rouge_l_f, bert_score_norm, float(intent_score)


def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score

def calculate_total_score(task_eval_groundtruth, task_eval_predict):
    import nltk
    s_bleu = nltk.translate.bleu_score.sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split())
    sim = nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict))
    from moverscore_v2 import get_idf_dict, word_mover_score 
    from collections import defaultdict
    mover = sentence_score(task_eval_predict, task_eval_groundtruth)
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(task_eval_predict, task_eval_groundtruth)
    rouge_f1 = scores[0]["rouge-1"]["f"]
    rouge_l_f1 = scores[0]["rouge-l"]["f"]
    from datasets import load_metric
    bertscore = load_metric("bertscore")
    bert_results = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased")
    bert_f1 = bert_results["f1"][0]
    bert_f1_norm = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased", lang="en", verbose=False, rescale_with_baseline=True)["f1"][0]
    return s_bleu, sim, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover


def get_metric_csv_line(model_type, task_eval_groundtruth, task_eval_predict, metric_intent_score):
    s_bleu, sim, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover = calculate_total_score(task_eval_groundtruth, task_eval_predict)
    avg_length = task_eval_predict.count("step") + task_eval_predict.count("Step")
    from icecream import ic
    csv_line = [model_type] + [s_bleu] + [sim] + [bert_f1] + [rouge_f1] + [bert_f1_norm] + [rouge_l_f1] + [mover] + metric_intent_score + [avg_length]
    return csv_line
