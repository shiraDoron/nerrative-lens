import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from config import NUM_TOPICS, NUM_NARRATIVES
from transformers import AutoModel
from peft import PeftModel
from sentence_transformers import models

# Stance learning
class TopicStanceLayer(nn.Module):
    def __init__(self):
        super(TopicStanceLayer, self).__init__()

        self.num_topics = NUM_TOPICS
        self.num_narratives = NUM_NARRATIVES

        # Opposed 0, neutral 1, supportive 2
        self.num_stances = 3

        # Vocabulary size = number of topics * 3
        self.matrix_size = self.num_topics * self.num_stances
        self.weights = nn.Embedding(self.matrix_size, self.num_narratives)

        # Initialization
        nn.init.uniform_(self.weights.weight, 0, 1)

    # Learning function
    # Receives topics and stances
    def forward(self, topic_ids, stance_labels):
        # Set the OOV index as the last topic (499)
        oov_topic_index = self.num_topics - 1

        # Safety: replace -1 or out-of-range values (above 499) with the OOV index
        # torch.clamp and torch.where help us stay within bounds
        safe_topic_ids = torch.where(
            (topic_ids == -1) | (topic_ids >= self.num_topics),
            torch.tensor(oov_topic_index, device=topic_ids.device),
            topic_ids
        )

        # Compute the composite index (between 0 and 1499)
        indices = (safe_topic_ids * self.num_stances) + stance_labels

        # Final safety check that the index doesn't exceed matrix_size (e.g. if stance_label is invalid)
        indices = torch.clamp(indices, 0, self.matrix_size - 1)

        # Fetch the matching vectors (the narrative percentages)
        narrative_vectors = self.weights(indices)

        return narrative_vectors


def load_merged_sentence_transformer(base_model_name, peft_model_id):
    base_model = AutoModel.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
    merged_model = peft_model.merge_and_unload()

    word_embedding_model = models.Transformer(base_model_name)
    word_embedding_model.auto_model = merged_model
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Running the stance detection models
class TopicAnalysisPipeline:
    def __init__(self):
        print("Loading Stance Model...")
        # Properly loading the base model + the fine-tuned adapter layer
        self.stance_model = load_merged_sentence_transformer(
            "sentence-transformers/all-mpnet-base-v2",
            "vahidthegreat/StanceAware-SBERT"
        )

        print("Loading pre-trained Topic Model...")
        # Loading the ready model from the folder we saved it to earlier
        self.topic_model = BERTopic.load("saved_topic_model")
        print("Pipeline is ready!")

    # Performs stance classification using anchor sentences
    def detect_stance(self, text, topic_text):
        anchor_pro = f"I support {topic_text}"
        anchor_con = f"I oppose {topic_text}"

        embeddings = self.stance_model.encode([text, anchor_pro, anchor_con])

        sim_pro = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_con = util.cos_sim(embeddings[0], embeddings[2]).item()

        diff = sim_pro - sim_con

        if diff > 0.02:
            return 2  # Support
        elif diff < -0.02:
            return 0  # Oppose
        else:
            return 1  # Neutral

    # Detects the main topic, and returns it along with the stance toward it
    def process_text(self, text):
        topics, probs = self.topic_model.transform([text])
        main_topic_id = topics[0]

        if main_topic_id == -1:
            return -1, 1

        topic_info = self.topic_model.get_topic(main_topic_id)

        if isinstance(topic_info, list) and len(topic_info) > 0:
            topic_name = topic_info[0][0]
        else:
            topic_name = "General"

        stance = self.detect_stance(text, topic_name)

        return main_topic_id, stance


# --- Test code ---
if __name__ == "__main__":
    print("Initializing models (this may take a few seconds)...")
    pipeline = TopicAnalysisPipeline()

    # Arbitrary data pool so BERTopic can generate "clusters"
    with open("gemini_natural_dataset.csv", "r", encoding="utf-8") as f:
        test_full_data = [line.strip() for line in f if line.strip()]

    # Test sentences we want to analyze against the generated clusters
    test_sentences = test_full_data[:10]

    print("\n--- Analysis Results ---")
    for sentence in test_sentences:
        topic_id, test_stance = pipeline.process_text(sentence, pipeline.topic_model)

        stance_text = {0: "Oppose", 1: "Neutral", 2: "Support"}.get(test_stance)
        test_topic_name = "Not detected"

        if topic_id != -1:
            test_topic_info = pipeline.topic_model.get_topic(topic_id)
            if test_topic_info:
                topic_name = test_topic_info[0][0]  # the strongest word

        print(f"Text: '{sentence}'")
        print(f"Detected topic: {test_topic_name} (ID: {topic_id})")
        print(f"Stance toward the topic: {stance_text} (code: {test_stance})\n")