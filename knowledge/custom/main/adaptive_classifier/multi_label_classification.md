# Multi-Label Adaptive Classification - Support Ticket Categorization

## Problem
Traditional single-label classifiers fail in real-world scenarios where text documents naturally belong to multiple categories simultaneously. Support tickets, product reviews, and content moderation require handling overlapping categories like "technical" + "urgent" + "security" rather than forcing single-label assignments. This limitation creates incomplete categorization and missed critical issues in production systems.

## Content
The MultiLabelClassifier solves multi-label classification through intelligent flattening of label combinations while maintaining the adaptive classifier's core benefits. Key innovations include:

1. **Label Set Flattening**: Transforms multi-label examples into single-label training pairs while preserving semantic relationships
2. **Threshold-based Filtering**: Uses configurable probability thresholds (0.3 default) and minimum probability differences (0.1 default) to determine relevant labels
3. **Adaptive Threshold Tuning**: Supports dynamic threshold adjustment based on domain requirements
4. **Cross-Category Learning**: Enables learning across overlapping categories like {"authentication", "technical", "bug"} simultaneously

The implementation demonstrates real-world application with 50 support ticket examples across 10 overlapping categories: authentication, payment, performance, UI/UX, technical bugs, data/privacy, feature requests, security issues, urgent matters, and integration problems.

## Usage
```python
class MultiLabelClassifier:
    def __init__(self, model_name="distilbert/distilbert-base-uncased", threshold=0.3, min_probability_diff=0.1):
        self.classifier = AdaptiveClassifier(model_name)
        self.threshold = threshold
        self.min_probability_diff = min_probability_diff
    
    def add_examples(self, texts, label_sets):
        # Handle multi-label examples
        flat_texts, flat_labels = self._flatten_multi_label(texts, label_sets)
        self.classifier.add_examples(flat_texts, flat_labels)
    
    def predict(self, text):
        predictions = self.classifier.predict(text)
        return [(label, prob) for label, prob in predictions 
                if prob >= self.threshold and meets_diff_criteria]

# Usage example
texts = ["Login page crashes during payment processing"]
labels = [{"authentication", "payment", "technical", "urgent"}]
classifier = MultiLabelClassifier()
classifier.add_examples(texts, labels)
predictions = classifier.predict("Cannot login and payment failed")
```

## Memory Relation
- **Previous**: [Advanced Usage Patterns](./advanced_usage_patterns.md)
- **Next**: [Multilingual Sentiment Analysis](./multilingual_sentiment_analysis.md)

## File Information
- **Path**: multi_label_classifier.py
- **Lines**: L1-200

## Tags
- multi_label_classification
- support_tickets
- production_systems
- threshold_filtering
- overlapping_categories
- adaptive_thresholds
- enterprise_categorization
