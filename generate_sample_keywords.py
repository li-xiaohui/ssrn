"""
Script to generate random sample keywords file for testing the topic extractor app.
Each line contains keywords for a paper, delimited by commas.
"""
import random

# Sample keyword sets for different research topics
keyword_sets = [
    ["deep learning", "natural language processing", "transformer", "attention mechanism", "neural networks", "sentiment analysis", "text classification"],
    ["reinforcement learning", "autonomous vehicles", "deep Q-networks", "navigation", "traffic control", "safety", "simulation"],
    ["federated learning", "privacy", "distributed systems", "machine learning", "aggregation", "non-IID data", "communication"],
    ["computer vision", "object detection", "convolutional neural networks", "multi-scale features", "attention", "real-time", "COCO dataset"],
    ["recommendation systems", "graph neural networks", "collaborative filtering", "embeddings", "social networks", "precision", "recall"],
    ["quantum machine learning", "variational quantum circuits", "quantum computing", "classification", "regression", "quantum simulators"],
    ["time series forecasting", "LSTM", "long short-term memory", "financial markets", "energy consumption", "prediction accuracy"],
    ["adversarial robustness", "deep neural networks", "image classification", "defense mechanisms", "adversarial attacks", "data augmentation", "CIFAR-10"],
    ["natural language processing", "knowledge graphs", "neural relation extraction", "transformer models", "entity linking", "relation prediction"],
    ["anomaly detection", "network traffic", "unsupervised learning", "autoencoders", "clustering", "network security", "intrusion detection"],
    ["transfer learning", "medical image analysis", "domain adaptation", "imaging modalities", "diagnostic accuracy", "medical imaging"],
    ["multi-modal learning", "vision", "language understanding", "visual encoders", "language models", "visual question answering", "image captioning"],
    ["distributed machine learning", "parameter server", "synchronization", "communication overhead", "scalability", "straggler nodes"],
    ["generative adversarial networks", "data augmentation", "low-resource settings", "synthetic data", "text classification", "image recognition"],
    ["explainable AI", "interpretability", "deep learning", "attention visualization", "feature importance", "model explanations", "trust"],
]

def generate_keywords_file(num_papers=15, output_file="sample_keywords.txt"):
    """Generate a file with random keywords, one line per paper."""
    selected_keywords = random.sample(keyword_sets, min(num_papers, len(keyword_sets)))
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for keywords in selected_keywords:
            # Join keywords with commas
            line = ', '.join(keywords)
            f.write(line + '\n')
    
    print(f"Generated {len(selected_keywords)} keyword sets and saved to {output_file}")
    return output_file

if __name__ == "__main__":
    import sys
    
    # Default: generate 15 papers
    num_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    
    print("Generating sample keywords file...")
    generate_keywords_file(num_papers, "sample_keywords.txt")
    print("\nDone! File created: sample_keywords.txt")
    print("Format: Each line contains keywords for a paper, separated by commas")

