import torch
import numpy as np
import datasets
from pathlib import Path

class ELPipeline:
    def __init__(self, bi_encoder, cross_encoder, bi_encoder_tokenizer, cross_encoder_tokenizer, embeddings, entity2description, wikipedia_mapper, wikidata_getter):
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        
        self.bi_encoder_tokenizer = bi_encoder_tokenizer
        self.cross_encoder_tokenizer = cross_encoder_tokenizer
        
        self.embeddings = embeddings

        self.entity2description = entity2description
        
        self.label2id = self.bi_encoder.config.label2id
        self.id2label = self.bi_encoder.config.id2label
        
        self.is_b_ids = torch.tensor([self.id2label[i].startswith('B-') for i in range(len(self.label2id))])
        self.is_i_ids = torch.tensor([self.id2label[i].startswith('I-') for i in range(len(self.label2id))])
        
        self.wikipedia_mapper = wikipedia_mapper
        
        self.wikidata_getter = wikidata_getter
        
    def compute_candidates(self, sentences, k=10):
        input_features = self.bi_encoder_tokenizer(sentences, return_tensors='pt', padding=True, return_special_tokens_mask=True, return_offsets_mapping=True)
        outputs = self.bi_encoder(mention_input_ids=input_features['input_ids'], mention_attention_mask=input_features['attention_mask'])
        
        ner_predictions = outputs.ner_logits.argmax(dim=-1)
        ner_label_ids = outputs.ner_logits.argsort(dim=-1, descending=True)[:, :, 0]
        special_tokens_mask = input_features['special_tokens_mask'] == 0
        
        is_b = self.is_b_ids[ner_label_ids] & special_tokens_mask
        # is_b = ((ner_predictions == self.bi_encoder.config.label2id['B']) & (input_features['special_tokens_mask'] == 0))
        
        is_i = self.is_i_ids[ner_label_ids] & special_tokens_mask
        # is_i = ((ner_predictions == self.bi_encoder.config.label2id['I']) & (input_features['special_tokens_mask'] == 0))
        
        mention_ids = torch.nonzero(is_b)
        
        mentions_input_ids = []
        char_starts = []
        char_ends = []
        for id in mention_ids:
            row = id[0].item()
            start = id[1].item()
            
            for end in range(start + 1, ner_predictions.size(-1)):
                if not is_i[row, end].item():
                    break
                
            m = input_features['input_ids'][row, start:end].tolist()
            char_starts.append(input_features['offset_mapping'][row, start, 0].item())
            char_ends.append(input_features['offset_mapping'][row, end-1, 1].item())
            mentions_input_ids.append(m)
        mentions = self.bi_encoder_tokenizer.batch_decode(mentions_input_ids, skip_special_tokens=True)
        
        mention_logits = outputs.mention_logits[mention_ids[:, 0], mention_ids[:, 1], :]
        ner_label_ids = outputs.ner_logits[mention_ids[:, 0], mention_ids[:, 1]].argmax(dim=-1).tolist()
        ner_labels = [
            self.get_ner_label(nid)
            for nid in ner_label_ids
        ]

        el_predictions = []
        wikidata_ids = []
        wikidata_properties = []
        for v in mention_logits:
            neighbors = self.embeddings.get_most_similar(v, n=k)
            if k == 1:
                wikipedia_title = neighbors[0][0]
                qid = self.wikipedia_mapper.wikipedia_mapper.title_to_wikidata_id(wikipedia_title)
                el_predictions.append(wikipedia_title)
                wikidata_ids.append(qid)
                wikidata_properties.append(self.wikidata_getter[qid])
            else:
                titles = [x[0] for x in neighbors]
                qids = [self.wikipedia_mapper.title_to_wikidata_id(x) for x in titles]
                props = [self.wikidata_getter[qid] for qid in qids]
                
                el_predictions.append(titles)
                wikidata_ids.append(qids)
                wikidata_properties.append(props)
                        
        sequence_ids = mention_ids[:, 0].tolist()
        res = [
            {'text': sentences[sid], 'predictions': [] }
            for sid in range(len(sentences))
        ]
        for sid, m, cs, ce, e, qid, props, ner in zip(sequence_ids, mentions, char_starts, char_ends, el_predictions, wikidata_ids, wikidata_properties, ner_labels):
            res[sid]['predictions'].append({
                'mention': m,
                'start': cs,
                'end': ce,
                'candidates': e,
                'wikidata': qid,
                'wikidata_properties': props,
                'label': ner,
            })
        return res
            
            
    def mark_mention(self, text, mention_start, mention_end):
        mention_context_max_length = 200
        half_length = mention_context_max_length // 2
               
        before_mention = text[:mention_start]
        after_mention = text[mention_end:]
        mention_str = text[mention_start:mention_end]
        
        i = max(0, len(before_mention) - half_length)
        j = min(len(after_mention), half_length)

        marked_text = before_mention[i:] + '<start>' + mention_str + '<end>' + after_mention[:j]
        
        return marked_text

    def get_candidate_description(self, candidate):
        return candidate + '<title>' + self.entity2description[candidate]
            
    def rank_candidates(self, text, mention_start, mention_end, candidates):
        marked_text = self.mark_mention(text, mention_start, mention_end)
        
        mentions_in_context = [marked_text] * len(candidates)
        candidate_descriptions = [self.get_candidate_description(x) for x in candidates]
        
        features = self.cross_encoder_tokenizer(text=mentions_in_context, text_pair=candidate_descriptions, padding=True, truncation=True, return_tensors='pt')
        cross_encoder_outputs = self.cross_encoder(**features)
        candidate_scores = cross_encoder_outputs.logits.squeeze().tolist()
        return candidate_scores
    
    def __call__(self, texts, k=10):
        with torch.no_grad():
            el_candidates = self.compute_candidates(texts, k=k)
            if self.cross_encoder is not None:
                for doc in el_candidates:
                    txt = doc['text']
                    for mention in doc['predictions']:
                        scores = self.rank_candidates(txt, mention['start'], mention['end'], mention['candidates'])
                        mention['scores'] = scores
                    
            return el_candidates
        
    def get_ner_label(self, id):
        lab = self.id2label[id]
        if lab.startswith('B-') or lab.startswith('I-'):
            lab = lab[2:]
        return lab

# TODO: conserver uniquement la fonction title_to_wikidata, puisque les autres ne sont pas utilisées
class WikipediaMapper:
    def __init__(self, pages, redirects):
        pages_without_qids = pages[['page_id', 'page_title', 'is_redirect']]
        pages_with_redirects = pages_without_qids.merge(redirects, left_on='page_title', right_on='source')
        title2qid = pages.loc[pages['wikidata_id'] != ''][['page_id', 'page_title', 'wikidata_id']].rename(columns={'page_id': 'target_id', 'page_title': 'target_title'})
        self.pages_with_redirects_and_qids = pages_with_redirects.merge(title2qid, left_on='target', right_on='target_title').drop(columns=['source', 'target'])
        self.target_pages_with_qids = self.pages_with_redirects_and_qids[~self.pages_with_redirects_and_qids['is_redirect']]
        
        self.title2id = { t: i for i, t in enumerate(self.pages_with_redirects_and_qids['page_title'])}
          
    def get_index(self, column, value, df=None):
        if df is None:
            df = self.pages_with_redirects_and_qids
        (i,) = np.where(df[column] == value)
        if len(i) == 1:
            return i.item()
        else:
            return None
    
    def title_to_wikidata_id(self, title):
        i = self.title2id.get(title, None)
        # i = self.get_index('page_title', title)
        if i is None:
            return None
        return self.pages_with_redirects_and_qids['wikidata_id'].iat[i]
    
    def title_to_wikipedia_id(self, title, follow_redirects=False):
        i = self.get_index('page_title', title)
        if i is None:
            return None
        col = 'target_id' if follow_redirects else 'page_id'
        return self.pages_with_redirects_and_qids[col].iat[i]
    
    def wikidata_id_to_title(self, qid):
        i = self.get_index('wikidata_id', qid, df=self.target_pages_with_qids)
        if i is None:
            return None
        return self.target_pages_with_qids['page_title'].iat[i]
    
    def wikidata_id_to_wikipedia_id(self, qid):
        i = self.get_index('wikidata_id', qid, df=self.target_pages_with_qids)
        if i is None:
            return None
        return self.target_pages_with_qids['page_id'].iat[i]
    
    def wikipedia_id_to_title(self, wikipedia_id, follow_redirects=False):
        i = self.get_index('page_id', wikipedia_id)
        if i is None:
            return None
        col = 'target_title' if follow_redirects else 'page_title'
        return self.pages_with_redirects_and_qids[col].iat[i]
    
    def wikipedia_id_to_wikidata_id(self, wikipedia_id):
        i = self.get_index('page_id', wikipedia_id)
        if i is None:
            return None
        return self.pages_with_redirects_and_qids['wikidata_id'].iat[i]

    
class WikidataPropertyGetter:
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def default(cls):
        import pandas
        dtype = {
            'qid': object,
            'label': object,
            'description': object,
            'osm': object,
            'geonames': object,
            'longitude': float,
            'latitude': float,
            'altitude': float,
        }
        df = datasets.load_dataset("gcaillaut/wikidata-geoloc-properties-20220907", split="train").to_pandas().set_index("qid", drop=False)
        return cls(df)
        
    def __getitem__(self, i):
        try:
            row = self.data.loc[i].to_dict()
            row['wikidata'] = i
            return row
        except KeyError:
            return {c: None for c in self.data.columns}
    
    def filter_rows(self, ids):
        self.data = self.data.filter(items=ids, axis=0)

def keep_best_candidates(tweets, min_score=0.7, only_geoloc=False):
    """Filtre les prédictions pour ne garder que la plus probable (selon le modèle).
    - min_score est le score minimal qu’une entité candidate doit avoir
    - only_geoloc permet de définir si on ne veut conserver que les prédictions liées à des entités géographiques
    """
    geoloc_labels = ('GEOLOC', 'FACILITY', 'TRANSPORT')
    filtered_predictions = []
    
    for tw in tweets:
        # En pratique, l’objet tweet contiendra sûrement son id, le texte du tweet, la date d’émission… qu’il faudra conserver
        filtered_predictions_for_current_tweet = []
        for p in tw['predictions']:
            if only_geoloc and p['label'] not in geoloc_labels:
                continue
            
            scores = np.array(p['scores'])
            ibest = scores.argmax()
            if scores[ibest] > min_score:
                best_pred = {
                    'mention': p['mention'],
                    'label': p['label'],
                    'start': p['start'],
                    'end': p['end'],
                    'entity': p['candidates'][ibest],
                    'wikidata': p['wikidata'][ibest],
                    'properties': p['wikidata_properties'][ibest],
                }
                filtered_predictions_for_current_tweet.append(best_pred)
        filtered_predictions.append(filtered_predictions_for_current_tweet)
    return filtered_predictions