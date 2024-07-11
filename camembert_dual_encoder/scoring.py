import math
import torch
from sklearn.metrics import precision_recall_fscore_support
from statistics import mean
import itertools
import torch.nn.functional as F


def validate_model(model, dataloader):
    ner_predicted = []
    ner_expected = []
    mention_predicted = []
    mention_expected = []
    entity_sim = []
    model_losses = {}

    for batch in dataloader:
        outputs = model(**batch)
        ner_logits = outputs.ner_logits
        mention_logits = outputs.mention_logits
        entity_logits = outputs.entity_logits
        losses = outputs.losses

        for loss_name, loss_value in losses.items():
            if loss_name not in model_losses:
                model_losses[loss_name] = []
            model_losses[loss_name].append(loss_value.item())

        flat_ner_logits = ner_logits.view(-1, ner_logits.size(-1))
        flat_ner_labels = batch['ner_labels'].view(-1)
        not_ignored = flat_ner_labels >= 0

        ner_predictions_for_batch = flat_ner_logits[not_ignored].argmax(-1)
        ner_predicted.extend(ner_predictions_for_batch.tolist())
        ner_expected.extend(flat_ner_labels[not_ignored].tolist())

        mention_predicted.extend((ner_predictions_for_batch == 0).int().tolist())
        mention_expected.extend((flat_ner_labels[not_ignored] == 0).int().tolist())

        embedding_dim = entity_logits.size(-1)
        # unbatched_el_labels = el_labels.view(-1)

        entity_rows = [x[0] for x in batch['entity_mappings']]
        entity_cols = [x[1] for x in batch['entity_mappings']]
        only_entity_logits = mention_logits[entity_rows, entity_cols, :]

        unbatched_entity_logits = entity_logits.view(-1, embedding_dim)
        if len(only_entity_logits) > 0:
            if model.config.mention_entity_similarity == 'dot':
                sims = torch.bmm(
                    only_entity_logits.view(-1, 1, embedding_dim),
                    unbatched_entity_logits.view(-1, embedding_dim, 1)
                )
            elif model.config.mention_entity_similarity == 'cos':
                sims = torch.bmm(
                    F.normalize(only_entity_logits).view(-1, 1, embedding_dim),
                    F.normalize(unbatched_entity_logits).view(-1, embedding_dim, 1)
                )
            entity_sim.append(sims.mean().item())

    mention_precision, mention_recall, mention_fscore, _support = precision_recall_fscore_support(mention_expected, mention_predicted, average="binary", pos_label=1)

    micro_precision, micro_recall, micro_fscore, _support = precision_recall_fscore_support(
        ner_expected, ner_predicted, average="micro")
    macro_precision, macro_recall, macro_fscore, _support = precision_recall_fscore_support(
        ner_expected, ner_predicted, average="macro")
    res = {
        "NER Micro Precision": micro_precision,
        "NER Micro Recall": micro_recall,
        "NER Micro FScore": micro_fscore,
        "NER Macro Precision": macro_precision,
        "NER Macro Recall": macro_recall,
        "NER Macro FScore": macro_fscore,
        "EL Similarity": mean(entity_sim) if len(entity_sim) > 0 else math.nan,
        "Mention Precision": mention_precision,
        "Mention Recall": mention_recall,
        "Mention FScore": mention_fscore,
    }
    for k, v in model_losses.items():
        res[f"Loss {k}"] = mean(v)
    return res


def eval_model(model, test_loader, embeddings):
    expected_ner = []
    predicted_ner = []
    expected_mention = []
    predicted_mention = []

    total = 0
    positives_1 = 0
    positives_5 = 0
    positives_10 = 0
    positives_100 = 0

    for batch in test_loader:
        ner_labels = batch.pop("ner_labels")
        el_labels = batch.pop("el_labels")
        entity_mappings = batch.pop("entity_mappings")
        
        outputs = model(**batch)
        ner_logits = outputs.ner_logits
        mention_logits = outputs.mention_logits

        flat_ner_labels = ner_labels.view(-1)
        not_ignored = flat_ner_labels >= 0
        flat_ner_labels = flat_ner_labels[not_ignored]

        ner_predictions = ner_logits.argmax(dim=-1)
        ner_predictions_for_batch = ner_predictions.view(-1)[not_ignored]
        id2label = model.mention_encoder.inner_model.config.id2label
        is_entity_detected = [
            id2label[ner_predictions[i, j].item()].startswith('B')
            for i, j in entity_mappings
        ]

        predicted_ner.extend(ner_predictions_for_batch.tolist())
        expected_ner.extend(flat_ner_labels.tolist())

        predicted_mention.extend((ner_predictions_for_batch == 0).int().tolist())
        expected_mention.extend((flat_ner_labels == 0).int().tolist())

        entity_rows = [x[0] for x in entity_mappings]
        entity_cols = [x[1] for x in entity_mappings]
        only_entity_logits = mention_logits[entity_rows, entity_cols, :]
        expected_entity_labels = itertools.chain.from_iterable(el_labels)

        for v, exp_qid, detected in zip(only_entity_logits, expected_entity_labels, is_entity_detected):
            # If the entity has not been detected by the model, skip
            if not detected:
                continue
            total += 1
            neighbors = embeddings.get_most_similar(v, n=100)
            predicted_qids = [x[0] for x in neighbors]
            if exp_qid == predicted_qids[0]:
                positives_1 += 1
                positives_5 += 1
                positives_10 += 1
                positives_100 += 1
            elif exp_qid in predicted_qids[1:5]:
                positives_5 += 1
                positives_10 += 1
                positives_100 += 1
            elif exp_qid in predicted_qids[5:10]:
                positives_10 += 1
                positives_100 += 1
            elif exp_qid in predicted_qids[10:100]:
                positives_100 += 1

    mention_precision, mention_recall, mention_fscore, _support = precision_recall_fscore_support(expected_mention, predicted_mention, average="binary", pos_label=1)

    ner_micro_precision, ner_micro_recall, ner_micro_fscore, _support = precision_recall_fscore_support(
        expected_ner, predicted_ner, average="micro")
    ner_macro_precision, ner_macro_recall, ner_macro_fscore, _support = precision_recall_fscore_support(
        expected_ner, predicted_ner, average="macro")

    el_r_at_1 = 0 if total == 0 else positives_1 / total
    el_r_at_5 = 0 if total == 0 else positives_5 / total
    el_r_at_10 = 0 if total == 0 else positives_10 / total
    el_r_at_100 = 0 if total == 0 else positives_100 / total

    res = {
        "NER Micro Precision": ner_micro_precision,
        "NER Micro Recall": ner_micro_recall,
        "NER Micro FScore": ner_micro_fscore,
        "NER Macro Precision": ner_macro_precision,
        "NER Macro Recall": ner_macro_recall,
        "NER Macro FScore": ner_macro_fscore,
        "Mention Precision": mention_precision,
        "Mention Recall": mention_recall,
        "Mention FScore": mention_fscore,
        "EL R@1": el_r_at_1,
        "EL R@5": el_r_at_5,
        "EL R@10": el_r_at_10,
        "EL R@100": el_r_at_100,
    }
    return res
