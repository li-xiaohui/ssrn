"""
Script to generate random sample abstracts for testing the topic extractor app.
"""
import random
import pandas as pd

# Sample abstracts covering different research topics
sample_abstracts = [
    "This paper presents a novel deep learning architecture for natural language processing tasks. We propose a transformer-based model that incorporates attention mechanisms to improve semantic understanding. Our approach achieves state-of-the-art performance on several benchmark datasets including GLUE and SuperGLUE. The model demonstrates significant improvements in tasks such as sentiment analysis, question answering, and text classification.",
    
    "We investigate the application of reinforcement learning algorithms to autonomous vehicle navigation. Our framework combines deep Q-networks with hierarchical decision-making to handle complex traffic scenarios. Experimental results show improved safety metrics and reduced collision rates compared to traditional control systems. The approach is validated through extensive simulations and real-world testing.",
    
    "This work explores federated learning techniques for privacy-preserving machine learning across distributed systems. We introduce a new aggregation algorithm that reduces communication overhead while maintaining model accuracy. Our method addresses challenges related to non-IID data distribution and heterogeneous network conditions. Evaluation on multiple datasets demonstrates competitive performance with centralized training approaches.",
    
    "We propose a new computer vision framework for object detection using convolutional neural networks. The architecture integrates multi-scale feature extraction with attention mechanisms to improve detection accuracy. Our method achieves superior performance on COCO and Pascal VOC datasets. The approach is computationally efficient and suitable for real-time applications.",
    
    "This paper addresses the problem of recommendation systems using graph neural networks. We develop a novel embedding technique that captures both user-item interactions and social network structures. The proposed method outperforms traditional collaborative filtering approaches on multiple recommendation benchmarks. Experimental results demonstrate improved precision and recall metrics.",
    
    "We present a comprehensive study on quantum machine learning algorithms and their applications. Our research explores variational quantum circuits for classification and regression tasks. We analyze the computational complexity and provide theoretical guarantees for convergence. The framework is tested on quantum simulators and demonstrates potential advantages over classical methods.",
    
    "This work introduces a new approach to time series forecasting using long short-term memory networks. We incorporate external factors such as weather data and economic indicators to improve prediction accuracy. The model is evaluated on financial market data and energy consumption datasets. Results show significant improvements over baseline methods in terms of mean absolute error.",
    
    "We investigate adversarial robustness in deep neural networks for image classification. Our research develops new defense mechanisms against adversarial attacks while maintaining clean accuracy. We propose a training strategy that enhances model resilience through data augmentation and regularization techniques. Evaluation on CIFAR-10 and ImageNet demonstrates improved robustness metrics.",
    
    "This paper explores the intersection of natural language processing and knowledge graphs. We develop a framework for extracting structured information from unstructured text using neural relation extraction. The approach combines transformer models with graph neural networks to improve entity linking and relation prediction. Experimental results on benchmark datasets show state-of-the-art performance.",
    
    "We propose a new method for anomaly detection in network traffic using unsupervised learning techniques. Our approach leverages autoencoders and clustering algorithms to identify suspicious patterns. The system is evaluated on real-world network datasets and demonstrates high detection rates with low false positive rates. The framework is scalable and suitable for production environments.",
    
    "This work presents a comprehensive analysis of transfer learning strategies for medical image analysis. We investigate domain adaptation techniques to improve model performance across different imaging modalities. Our approach addresses challenges related to limited labeled data and domain shift. Evaluation on medical imaging datasets shows improved diagnostic accuracy.",
    
    "We develop a new framework for multi-modal learning that combines vision and language understanding. The architecture integrates visual encoders with language models to enable cross-modal reasoning. Our method achieves competitive results on visual question answering and image captioning tasks. The approach demonstrates improved understanding of complex visual scenes.",
    
    "This paper addresses challenges in distributed machine learning systems. We propose a new synchronization protocol that reduces communication overhead in parameter server architectures. Our method handles straggler nodes and network failures gracefully. Experimental evaluation shows improved training efficiency and scalability.",
    
    "We investigate the application of generative adversarial networks for data augmentation in low-resource settings. Our approach generates synthetic training examples that improve model generalization. The method is evaluated on text classification and image recognition tasks. Results demonstrate significant improvements when training data is limited.",
    
    "This work explores explainable artificial intelligence techniques for deep learning models. We develop new methods for generating interpretable explanations of model predictions. Our approach combines attention visualization with feature importance analysis. Evaluation on multiple domains shows improved user trust and understanding of model decisions.",
]

def generate_random_abstracts_file(num_abstracts=15, output_file="sample_abstracts.txt"):
    """Generate a file with random abstracts."""
    selected_abstracts = random.sample(sample_abstracts, min(num_abstracts, len(sample_abstracts)))
    
    # Save as text file (one abstract per line, separated by double newlines)
    with open(output_file, 'w', encoding='utf-8') as f:
        for abstract in selected_abstracts:
            f.write(abstract + '\n\n')
    
    print(f"Generated {len(selected_abstracts)} abstracts and saved to {output_file}")
    return output_file

def generate_random_abstracts_csv(num_abstracts=15, output_file="sample_abstracts.csv"):
    """Generate a CSV file with random abstracts."""
    selected_abstracts = random.sample(sample_abstracts, min(num_abstracts, len(sample_abstracts)))
    
    df = pd.DataFrame({
        'abstract': selected_abstracts,
        'paper_id': [f"paper_{i+1}" for i in range(len(selected_abstracts))]
    })
    
    df.to_csv(output_file, index=False)
    print(f"Generated {len(selected_abstracts)} abstracts and saved to {output_file}")
    return output_file

if __name__ == "__main__":
    import sys
    
    # Default: generate both formats
    num_abstracts = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    
    print("Generating sample abstracts...")
    generate_random_abstracts_file(num_abstracts, "sample_abstracts.txt")
    generate_random_abstracts_csv(num_abstracts, "sample_abstracts.csv")
    print("\nDone! Files created:")
    print("  - sample_abstracts.txt")
    print("  - sample_abstracts.csv")

