#import torch
#import torch.nn as nn
#import json
#from bertopic import BERTopic
#from tqdm import tqdm
#import transformers
#import logging
#
## import all the components we built, from the various files
#from ner import NarrativeEntityLayer, EntityAnalysisPipeline
#from srl import SRLProcessor, SRLNarrativeLayer
#from emotion import EmotionAgencyProcessor, EmotionAgencyLayer
#from reliability import ReliabilityProcessor, ReliabilityLayer
#from stance import TopicStanceLayer, TopicAnalysisPipeline
#from config import NUM_NARRATIVES, NARRATIVES
#
#transformers.logging.set_verbosity_error()
#logging.getLogger("transformers").setLevel(logging.ERROR)
#
#
## MLP network
#class NarrativeFusionNetwork(nn.Module):
#    """
#    Upgraded fusion network (MLP): concatenates the vectors from every model,
#    learns non-linear relationships between them, and applies the reliability
#    factor to produce the final decision.
#    """
#
#    def __init__(self):
#        super(NarrativeFusionNetwork, self).__init__()
#
#        # We receive 4 vectors, each sized by the number of narratives
#        input_size = 4 * NUM_NARRATIVES
#
#        # Hidden layer size
#        hidden_size = 64
#
#        # Building the neural network
#        self.mlp = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU(),
#            nn.Dropout(0.3),
#            nn.Linear(hidden_size, NUM_NARRATIVES)
#        )
#
#        self.softmax = nn.Softmax(dim=-1)
#
#    def forward(self, vec_ner, vec_stance, vec_srl, vec_emotion, weight_factor):
#        # Concatenate all vectors into a single tensor
#        combined_features = torch.cat(
#            [vec_ner.squeeze(), vec_stance.squeeze(), vec_srl.squeeze(), vec_emotion.squeeze()], dim=-1)
#
#        # Pass through the learned neural network
#        fused_logits = self.mlp(combined_features)
#
#        # Apply the reliability factor
#        amplified_signal = fused_logits * weight_factor
#
#        # Convert to probabilities
#        return self.softmax(amplified_signal)
#
#
#class NarrativeDetector(nn.Module):
#    """
#    The full Pipeline that packages all the learned and processing components.
#    """
#
#    def __init__(self, ner_vocab, srl_vocab):
#        super(NarrativeDetector, self).__init__()
#
#        print("Initializing Full Narrative Detection Pipeline...")
#
#        pbar = tqdm(total=5, desc="Loading Models",
#                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
#
#        self.ner_processor = EntityAnalysisPipeline()
#        pbar.update(1)
#        self.srl_processor = SRLProcessor()
#        pbar.update(1)
#        self.emotion_processor = EmotionAgencyProcessor()
#        pbar.update(1)
#        self.stance_processor = TopicAnalysisPipeline()  # the updated topics module
#        pbar.update(1)
#        self.reliability_processor = ReliabilityProcessor()
#        pbar.update(1)
#
#        pbar.close()
#
#        # The learned layers inside PyTorch
#        self.ner_layer = NarrativeEntityLayer(ner_vocab)
#        self.srl_layer = SRLNarrativeLayer(srl_vocab)
#        self.emotion_layer = EmotionAgencyLayer()
#        self.stance_layer = TopicStanceLayer()  # the layer that now only ingests topics
#        self.reliability_layer = ReliabilityLayer()
#
#        # The fusion network
#        self.fusion_network = NarrativeFusionNetwork()
#
#        self.ner_vocab = ner_vocab
#        self.srl_vocab = srl_vocab
#
#    def extract_features(self, text):
#        return {
#            "ner": self.ner_processor.extract_entities(text, self.ner_vocab),
#            "srl": self.srl_processor.extract_features(text, self.srl_vocab),
#            "emotion": self.emotion_processor.extract_features(text),
#            "stance": self.stance_processor.process_text(text),  # now returns only topic_id
#            "reliability": self.reliability_processor.extract_features(text)
#        }
#
#    def classify_features(self, features):
#        vec_ner = self.ner_layer(features["ner"])
#        vec_srl = self.srl_layer(features["srl"])
#
#        emotion_idx, agency_flag = features["emotion"]
#        vec_emotion = self.emotion_layer(emotion_idx, agency_flag)
#
#        # --- the key fix to prevent a crash ---
#        topic_id = features["stance"]  # get only the topic id (no stance_label)
#        topic_tensor = torch.tensor([topic_id], dtype=torch.long)
#
#        # pass only the topic_tensor to the updated layer
#        vec_stance = self.stance_layer(topic_tensor)
#
#        weight_factor = self.reliability_layer(features["reliability"])
#
#        final_probs = self.fusion_network(vec_ner, vec_stance, vec_srl, vec_emotion, weight_factor)
#        return final_probs
#
#    def forward(self, text):
#        features = self.extract_features(text)
#        return self.classify_features(features)
#
#
## --- system-run simulation ---
#if __name__ == "__main__":
#    with open("shared_vocab.json", "r", encoding="utf-8") as f:
#        shared_vocab = json.load(f)
#
#    detector = NarrativeDetector(ner_vocab=shared_vocab, srl_vocab=shared_vocab)
#    detector.load_state_dict(torch.load("best_narrative_model_hybrid.pth"))
#
#    # fix the BERTopic model filename to match the one used by the training code
#    loaded_topic_model = BERTopic.load("saved_topic_model")
#    detector.stance_processor.topic_model = loaded_topic_model
#    detector.eval()
#
#    # example sentence
#    sample_text = "In my opinion, the UN might condemn the attack to protect the civilians."
#
#    print(f"\nAnalyzing Text: '{sample_text}'")
#    with torch.no_grad():
#        results = detector(sample_text).squeeze()
#
#    print("\nFinal Narrative Probabilities (Top 5):")
#    top_probs, top_indices = torch.topk(results, 5)
#
#    for i in range(5):
#        print(f"Narrative {NARRATIVES[top_indices[i].item()]}: {top_probs[i].item() * 100:.2f}%")

import torch
import torch.nn as nn
import json
from bertopic import BERTopic
from tqdm import tqdm
import transformers
import logging

# import all the components we built, from the various files
from ner import NarrativeEntityLayer, EntityAnalysisPipeline
from srl import SRLProcessor, SRLNarrativeLayer
from emotion import EmotionAgencyProcessor, EmotionAgencyLayer
from reliability import ReliabilityProcessor, ReliabilityLayer
from stance import TopicStanceLayer, TopicAnalysisPipeline  # now used only as a topics module
from config import NUM_NARRATIVES, NARRATIVES

# extra components for Hybrid v1 (SBERT + engineered features, see bottom of file)
from sentence_transformers import SentenceTransformer
from analyze_agendas import AGENDA_PATTERNS, IDEOLOGY_PATTERNS, clean_text as clean_agenda_text

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


# =====================================================================
# Linear fusion network (Linear Fusion Network) for transparency and disentangling module weights
# =====================================================================
class NarrativeFusionNetwork(nn.Module):
    """
    Linear fusion network for testing module importance:
    learns a single weight for each module (NER, Topics, SRL, Emotion)
    and computes their weighted sum, enabling transparent interpretation of
    each component's importance.
    """

    def __init__(self):
        super(NarrativeFusionNetwork, self).__init__()

        # define 4 learned weights (one per module), initialized to an equal value
        self.module_weights = nn.Parameter(torch.ones(4))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, vec_ner, vec_topics, vec_srl, vec_emotion, weight_factor):
        # normalize the weights via Softmax so they always sum to 1 (100%)
        weights = torch.softmax(self.module_weights, dim=0)

        # compute the weighted sum: each vector is multiplied by the specific weight the model learned for it
        fused_logits = (weights[0] * vec_ner.squeeze() +
                        weights[1] * vec_topics.squeeze() +
                        weights[2] * vec_srl.squeeze() +
                        weights[3] * vec_emotion.squeeze())

        # apply the reliability factor to the weighted result
        amplified_signal = fused_logits * weight_factor

        # convert to probabilities
        return self.softmax(amplified_signal)


# =====================================================================
# The full Pipeline adapted to a topics-only module
# =====================================================================
class NarrativeDetector(nn.Module):
    """
    The full Pipeline that packages all the learned and processing components.
    Adapted to a topics-only module (no stance component).
    """

    def __init__(self, ner_vocab, srl_vocab):
        super(NarrativeDetector, self).__init__()

        print("Initializing Full Narrative Detection Pipeline (Topics Only)...")

        # feature extractors - not learned (heavy)
        pbar = tqdm(total=5, desc="Loading Models",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        self.ner_processor = EntityAnalysisPipeline()
        pbar.update(1)
        self.srl_processor = SRLProcessor()
        pbar.update(1)
        self.emotion_processor = EmotionAgencyProcessor()
        pbar.update(1)
        self.stance_processor = TopicAnalysisPipeline()
        pbar.update(1)
        self.reliability_processor = ReliabilityProcessor()
        pbar.update(1)

        pbar.close()

        # the learned layers inside PyTorch (fast)
        self.ner_layer = NarrativeEntityLayer(ner_vocab)
        self.srl_layer = SRLNarrativeLayer(srl_vocab)
        self.emotion_layer = EmotionAgencyLayer()
        self.stance_layer = TopicStanceLayer()  # now behaves as a Topic Layer only
        self.reliability_layer = ReliabilityLayer()

        # linear fusion network, for testing weights
        self.fusion_network = NarrativeFusionNetwork()

        self.ner_vocab = ner_vocab
        self.srl_vocab = srl_vocab

    def extract_features(self, text):
        return {
            "ner": self.ner_processor.extract_entities(text, self.ner_vocab),
            "srl": self.srl_processor.extract_features(text, self.srl_vocab),
            "emotion": self.emotion_processor.extract_features(text),
            "stance": self.stance_processor.process_text(text),  # returns a tuple of (topic_id, stance_label)
            "reliability": self.reliability_processor.extract_features(text)
        }

    def classify_features(self, features):
        # 1. process the entities/targets module
        vec_ner = self.ner_layer(features["ner"])
        vec_srl = self.srl_layer(features["srl"])

        # 2. process the patterns module (emotion and activity)
        emotion_idx, agency_flag = features["emotion"]
        vec_emotion = self.emotion_layer(emotion_idx, agency_flag)

        # 3. adaptation to a topics-only module: send only the topic_id to the learned layer
        topic_id = features["stance"]  # deliberately ignore stance_label (noise reduction)
        topic_tensor = torch.tensor([topic_id], dtype=torch.long)

        # note: the original stance_layer is applied here only to the topic tensor
        vec_topics = self.stance_layer(topic_tensor)

        # 4. extract the reliability factor
        weight_factor = self.reliability_layer(features["reliability"])

        # 5. weighted linear fusion
        final_probs = self.fusion_network(vec_ner, vec_topics, vec_srl, vec_emotion, weight_factor)
        return final_probs

    def forward(self, text):
        features = self.extract_features(text)
        return self.classify_features(features)


# =====================================================================
# Hybrid v1: shared SBERT embedding + explicit narrative features (concat + MLP)
# =====================================================================
# The three models compared in the thesis:
#   Baseline 1 : SBERT -> Narrative (only, no engineered features)
#   Baseline 2 : NER/SRL/Emotion/Stance/Reliability -> Fusion -> Narrative
#                (this is the existing NarrativeDetector above)
#   Proposed   : HybridNarrativeDetector - concatenates a shared semantic embedding (SBERT)
#   Hybrid       together with all the existing explicit features (NER, SRL, Emotion,
#                Reliability, Stance/Topics, Agenda, Ideology), and passes them through an MLP
#                to learn the final classification. Doesn't discard any existing work - extends it.
class AgendaIdeologyFeatureExtractor:
    """
    Extracts lexicon-based features (no learned component, no network): a binary vector
    indicating which agenda categories (AGENDA_LEXICON) and which ideology categories
    (IDEOLOGY_LEXICON) from analyze_agendas.py appear in the given text. Used as
    additional "engineered" features in the Hybrid model, alongside NER/SRL/Emotion/Stance.
    The output size is derived dynamically from the number of categories in the lexicons
    (not hardcoded), so it stays valid even if the lexicons in analyze_agendas.py change
    in the future.
    """

    def __init__(self):
        self.agenda_categories = list(AGENDA_PATTERNS.keys())
        self.ideology_categories = list(IDEOLOGY_PATTERNS.keys())
        self.output_size = len(self.agenda_categories) + len(self.ideology_categories)

    def extract_features(self, text):
        cleaned = clean_agenda_text(text)
        agenda_hits = [1.0 if AGENDA_PATTERNS[cat].search(cleaned) else 0.0 for cat in self.agenda_categories]
        ideology_hits = [1.0 if IDEOLOGY_PATTERNS[cat].search(cleaned) else 0.0 for cat in self.ideology_categories]
        return torch.tensor(agenda_hits + ideology_hits, dtype=torch.float32)


class HybridNarrativeDetector(nn.Module):
    """
    Hybrid v1 ("Proposed Hybrid"): combines a shared semantic embedding from SBERT with
    all the engineered features already existing in the project (NER, SRL, Emotion,
    Reliability, Stance/Topics, Agenda, Ideology) via concatenation, and passes them
    through an MLP to learn the final narrative classification.

    At this stage the SBERT encoder is frozen (not updated during training) - to keep a
    clean comparison against Baseline 2 (NarrativeDetector) and avoid overfitting on a
    small dataset; fine-tuning the encoder itself could be a future improvement once this
    basic Hybrid is proven useful.
    """

    def __init__(self, ner_vocab, srl_vocab, sbert_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 hidden_size=128, dropout=0.3):
        super(HybridNarrativeDetector, self).__init__()

        print("Initializing Hybrid Narrative Detector (SBERT + engineered features)...")

        pbar = tqdm(total=6, desc="Loading Models",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        # 1. shared semantic Encoder (frozen in v1)
        self.sbert = SentenceTransformer(sbert_model_name)
        pbar.update(1)

        # 2. engineered feature extractors - not learned (heavy), same as in NarrativeDetector
        self.ner_processor = EntityAnalysisPipeline()
        pbar.update(1)
        self.srl_processor = SRLProcessor()
        pbar.update(1)
        self.emotion_processor = EmotionAgencyProcessor()
        pbar.update(1)
        self.stance_processor = TopicAnalysisPipeline()
        pbar.update(1)
        self.reliability_processor = ReliabilityProcessor()
        pbar.update(1)
        pbar.close()

        # 3. agenda/ideology feature extractor (lexicon-based, no learning)
        self.agenda_ideology_processor = AgendaIdeologyFeatureExtractor()

        # 4. the existing learned layers (each produces a vector sized NUM_NARRATIVES)
        self.ner_layer = NarrativeEntityLayer(ner_vocab)
        self.srl_layer = SRLNarrativeLayer(srl_vocab)
        self.emotion_layer = EmotionAgencyLayer()
        self.stance_layer = TopicStanceLayer()
        self.reliability_layer = ReliabilityLayer()

        self.ner_vocab = ner_vocab
        self.srl_vocab = srl_vocab

        # 5. the MLP that combines everything: semantic embedding + all engineered features concatenated
        sbert_dim = self.sbert.get_sentence_embedding_dimension()
        # NER + SRL + Emotion + Stance: a vector sized NUM_NARRATIVES each,
        # + a single reliability scalar, + an agenda/ideology vector of dynamic size
        engineered_dim = (NUM_NARRATIVES * 4) + 1 + self.agenda_ideology_processor.output_size
        combined_dim = sbert_dim + engineered_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_NARRATIVES)
        )
        self.softmax = nn.Softmax(dim=-1)

    def extract_features(self, text):
        # shared semantic embedding from SBERT (frozen in v1, no gradients) - computed here
        # (like the other heavy features) so train.py can cache it together with the rest of
        # the features, instead of recomputing it every training epoch.
        with torch.no_grad():
            sbert_embedding = torch.tensor(
                self.sbert.encode(text, convert_to_numpy=True), dtype=torch.float32
            )

        return {
            "text": text,
            "sbert_embedding": sbert_embedding,
            "ner": self.ner_processor.extract_entities(text, self.ner_vocab),
            "srl": self.srl_processor.extract_features(text, self.srl_vocab),
            "emotion": self.emotion_processor.extract_features(text),
            "stance": self.stance_processor.process_text(text),
            "reliability": self.reliability_processor.extract_features(text),
            "agenda_ideology": self.agenda_ideology_processor.extract_features(text),
        }

    def classify_features(self, features):
        # engineered features (same as in the existing NarrativeDetector)
        vec_ner = self.ner_layer(features["ner"]).squeeze()
        vec_srl = self.srl_layer(features["srl"]).squeeze()

        emotion_idx, agency_flag = features["emotion"]
        vec_emotion = self.emotion_layer(emotion_idx, agency_flag).squeeze()

        topic_id = features["stance"]
        topic_tensor = torch.tensor([topic_id], dtype=torch.long)
        vec_topics = self.stance_layer(topic_tensor).squeeze()

        # reliability factor as a feature (a size-1 vector) - not as a multiplier like in Baseline 2
        reliability_vec = self.reliability_layer(features["reliability"])

        agenda_ideology_vec = features["agenda_ideology"]

        engineered = torch.cat(
            [vec_ner, vec_srl, vec_emotion, vec_topics, reliability_vec, agenda_ideology_vec], dim=-1
        )

        # shared semantic embedding - retrieved ready-made (already computed in extract_features)
        semantic_embedding = features["sbert_embedding"]

        combined = torch.cat([semantic_embedding, engineered], dim=-1)
        logits = self.mlp(combined)
        return self.softmax(logits)

    def forward(self, text):
        features = self.extract_features(text)
        return self.classify_features(features)


class SBERTOnlyDetector(nn.Module):
    """
    Baseline 1 ("SBERT-only"): a frozen SBERT embedding -> MLP -> narrative, without any
    additional engineered feature (NER/SRL/Emotion/Stance/Reliability/Agenda/Ideology). Used
    as a bottom line for the thesis's research comparison: how much the existing engineered
    features in HybridNarrativeDetector add on top of this baseline.
    Keeps the exact same structure (extract_features/classify_features/forward) as the other
    models in this file, so train.py can train it generically.
    """

    def __init__(self, sbert_model_name="sentence-transformers/all-MiniLM-L6-v2", hidden_size=128, dropout=0.3):
        super(SBERTOnlyDetector, self).__init__()

        print("Initializing SBERT-Only Narrative Detector (baseline)...")
        self.sbert = SentenceTransformer(sbert_model_name)

        sbert_dim = self.sbert.get_sentence_embedding_dimension()
        self.mlp = nn.Sequential(
            nn.Linear(sbert_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, NUM_NARRATIVES)
        )
        self.softmax = nn.Softmax(dim=-1)

    def extract_features(self, text):
        with torch.no_grad():
            sbert_embedding = torch.tensor(
                self.sbert.encode(text, convert_to_numpy=True), dtype=torch.float32
            )
        return {"sbert_embedding": sbert_embedding}

    def classify_features(self, features):
        logits = self.mlp(features["sbert_embedding"])
        return self.softmax(logits)

    def forward(self, text):
        features = self.extract_features(text)
        return self.classify_features(features)


# --- system-run simulation ---
if __name__ == "__main__":
    # load the model data
    with open("data/cache/shared_vocab.json", "r", encoding="utf-8") as f:
        shared_vocab = json.load(f)

    detector = NarrativeDetector(ner_vocab=shared_vocab, srl_vocab=shared_vocab)
    detector.load_state_dict(torch.load("models/best_narrative_model_hybrid.pth"))

    loaded_topic_model = BERTopic.load("models/best_bertopic_model")
    detector.stance_processor.topic_model = loaded_topic_model
    detector.eval()

    # example sentence
    sample_text = "In my opinion, the UN might condemn the attack to protect the civilians."

    print(f"\nAnalyzing Text: '{sample_text}'")
    with torch.no_grad():
        results = detector(sample_text).squeeze()

    print("\nFinal Narrative Probabilities (Top 5):")
    top_probs, top_indices = torch.topk(results, 5)

    for i in range(5):
        print(f"Narrative {NARRATIVES[top_indices[i].item()]}: {top_probs[i].item() * 100:.2f}%")