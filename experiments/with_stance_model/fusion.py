import torch
import torch.nn as nn
import json
from bertopic import BERTopic
from tqdm import tqdm
import transformers
import logging

# Import all the components we built across the various files
from ner import NarrativeEntityLayer, EntityAnalysisPipeline
from srl import SRLProcessor, SRLNarrativeLayer
from emotion import EmotionAgencyProcessor, EmotionAgencyLayer
from reliability import ReliabilityProcessor, ReliabilityLayer
from stance import TopicStanceLayer, TopicAnalysisPipeline
from config import NARRATIVES

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# MLP network
#class NarrativeFusionNetwork(nn.Module):
#    """
#    The upgraded fusion network (MLP): concatenates the vectors from every model,
#    learns non-linear relationships between them, and applies the reliability
#    factor to reach a final decision.
#    """
#
#    def __init__(self):
#        super(NarrativeFusionNetwork, self).__init__()
#
#        # We receive 4 vectors, each sized to the number of narratives
#        input_size = 4 * NUM_NARRATIVES
#
#        # Size of the hidden layer (can be changed as needed, 64 is a good starting point)
#        hidden_size = 64
#
#        # Building the neural network
#        self.mlp = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU(),  # non-linear activation function
#            nn.Dropout(0.3),  # randomly disable 30% of neurons to prevent overfitting
#            nn.Linear(hidden_size, NUM_NARRATIVES)  # final output sized to the number of narratives
#        )
#
#        self.softmax = nn.Softmax(dim=-1)
#
#    def forward(self, vec_ner, vec_stance, vec_srl, vec_emotion, weight_factor):
#        # 1. Concatenate all vectors into a single tensor (e.g. 4 * 17 = 68 dimensions)
#        combined_features = torch.cat([vec_ner.squeeze(), vec_stance.squeeze(), vec_srl.squeeze(), vec_emotion.squeeze()], dim=-1)
#
#        # 2. Pass through the learned neural network
#        fused_logits = self.mlp(combined_features)
#
#        # 3. Apply the reliability factor (multiply the network's output by the factor)
#        amplified_signal = fused_logits * weight_factor
#
#        # 4. Convert to probabilities
#        return self.softmax(amplified_signal)

# Linear network
class NarrativeFusionNetwork(nn.Module):
    """
    Linear fusion network for testing module importance:
    learns a single weight for each module (NER, Stance, SRL, Emotion)
    and computes their weighted sum, allowing a transparent interpretation of
    each component's importance.
    """

    def __init__(self):
        super(NarrativeFusionNetwork, self).__init__()

        # Define 4 learned weights (one per module), initialized to equal values
        self.module_weights = nn.Parameter(torch.ones(4))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, vec_ner, vec_stance, vec_srl, vec_emotion, weight_factor):
        # Normalize the weights so they always sum to 1 (100%), making them easy to interpret
        weights = torch.softmax(self.module_weights, dim=0)

        # Compute the weighted sum: each vector is multiplied by the specific weight the model learned for it
        fused_logits = (weights[0] * vec_ner.squeeze() +
                        weights[1] * vec_stance.squeeze() +
                        weights[2] * vec_srl.squeeze() +
                        weights[3] * vec_emotion.squeeze())

        # Apply the reliability factor to the weighted result
        amplified_signal = fused_logits * weight_factor

        # Convert to probabilities
        return self.softmax(amplified_signal)


class NarrativeDetector(nn.Module):
    """
    The full pipeline that wraps all the learned components and processors.
    """

    def __init__(self, ner_vocab, srl_vocab):
        super(NarrativeDetector, self).__init__()

        print("Initializing Full Narrative Detection Pipeline...")

        # Feature extractors - not learned (heavy)
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

        # The learned layers within PyTorch (fast)
        self.ner_layer = NarrativeEntityLayer(ner_vocab)
        self.srl_layer = SRLNarrativeLayer(srl_vocab)
        self.emotion_layer = EmotionAgencyLayer()
        self.stance_layer = TopicStanceLayer()
        self.reliability_layer = ReliabilityLayer()

        # The updated fusion network
        self.fusion_network = NarrativeFusionNetwork()

        self.ner_vocab = ner_vocab
        self.srl_vocab = srl_vocab

    def extract_features(self, text):
        return {
            "ner": self.ner_processor.extract_entities(text, self.ner_vocab),
            "srl": self.srl_processor.extract_features(text, self.srl_vocab),
            "emotion": self.emotion_processor.extract_features(text),
            "stance": self.stance_processor.process_text(text),
            "reliability": self.reliability_processor.extract_features(text)
        }

    def classify_features(self, features):
        vec_ner = self.ner_layer(features["ner"])
        vec_srl = self.srl_layer(features["srl"])

        emotion_idx, agency_flag = features["emotion"]
        vec_emotion = self.emotion_layer(emotion_idx, agency_flag)

        topic_id, stance_label = features["stance"]
        topic_tensor = torch.tensor([topic_id], dtype=torch.long)
        stance_tensor = torch.tensor([stance_label], dtype=torch.long)
        vec_stance = self.stance_layer(topic_tensor, stance_tensor)

        weight_factor = self.reliability_layer(features["reliability"])

        final_probs = self.fusion_network(vec_ner, vec_stance, vec_srl, vec_emotion, weight_factor)
        return final_probs

    def forward(self, text):
        features = self.extract_features(text)
        return self.classify_features(features)


# --- System run simulation ---
if __name__ == "__main__":
    # Load the model data
    with open("../shared_vocab.json", "r", encoding="utf-8") as f:
        shared_vocab = json.load(f)
    detector = NarrativeDetector(ner_vocab=shared_vocab, srl_vocab=shared_vocab)
    detector.load_state_dict(torch.load("ללא התיקון/best_narrative_model_hybrid.pth"))
    loaded_topic_model = BERTopic.load("best_bertopic_model")
    detector.stance_processor.topic_model = loaded_topic_model
    detector.eval()

    # Sample sentence
    sample_text = "In my opinion, the UN might condemn the attack to protect the civilians."

    print(f"\nAnalyzing Text: '{sample_text}'")
    with torch.no_grad():
        results = detector(sample_text).squeeze()

    print("\nFinal Narrative Probabilities (Top 5):")
    top_probs, top_indices = torch.topk(results, 5)

    for i in range(5):
        print(f"Narrative {NARRATIVES[top_indices[i].item()]}: {top_probs[i].item() * 100:.2f}%")