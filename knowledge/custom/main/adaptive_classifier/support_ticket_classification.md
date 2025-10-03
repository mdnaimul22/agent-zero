# Support Ticket Classification - Insurance Customer Service Automation

## Problem
Insurance companies face massive volumes of customer support tickets requiring immediate triage and routing to appropriate departments. Traditional rule-based systems fail to understand the nuanced differences between policy inquiries, billing issues, and claim-related questions. Manual ticket categorization creates bottlenecks and inconsistent customer experiences, while automated systems struggle with the semantic complexity of insurance-specific language and customer intent.

## Content
The support ticket classification system provides a comprehensive solution for insurance customer service automation through adaptive classification of customer conversations. Key innovations include:

1. **Intent-Based Classification**: Distinguishes between policy details, account/billing issues, and claim-related inquiries using semantic understanding of insurance terminology
2. **Conversational Context Analysis**: Analyzes entire customer-bot conversations rather than isolated messages for accurate intent classification
3. **Insurance-Specific Training**: Uses real insurance customer service examples with proper handling of policy terms, coverage details, and billing terminology
4. **Continuous Learning Pipeline**: Improves accuracy over time by learning from new customer interactions and edge cases

The system demonstrates with 50+ real insurance customer service examples covering policy coverage questions, billing disputes, payment method changes, coverage explanations, and account management issues.

## Usage
```python
# Initialize support ticket classifier
classifier = AdaptiveClassifier("google-bert/bert-large-cased")

# Define insurance-specific classes with detailed descriptions
classes = {
    "policy_details": """Chats about policy specifics like coverage, limits, and exclusions.
    Examples include questions about:
    - Policy coverage details and exclusions
    - Coverage limits and deductibles
    - Policy duration and renewal terms
    - Specific policy features and benefits
    - Types of insurance policies available""",

    "account_and_billing": """Chat messages about payments, billing issues, and account management.
    Examples include:
    - Payment processing and autopay setup
    - Bill explanations and charge disputes
    - Account balance inquiries
    - Payment method updates
    - Late payment and fee questions"""
}

# Training examples with full conversation context
examples = {
    "policy_details": [
        """User: What does my homeowner's insurance cover?
        Bot: Our standard homeowner's insurance covers structural damage, personal property, liability, and additional living expenses.
        User: Tell me more about the structural damage coverage.
        Bot: Structural damage coverage protects against damages from events like fire, storms, or vandalism."""
    ],

    "account_and_billing": [
        """User: I want to change my payment date.
        Bot: I can help with that. When would you prefer to make your payments?
        User: Can I move it to the 15th of each month?
        Bot: Yes, we can adjust your payment date to the 15th."""
    ]
}

# Train classifier with descriptions and examples
for class_name, description in classes.items():
    classifier.add_examples([description], [class_name])
    classifier.add_examples(examples[class_name], [class_name] * len(examples[class_name]))

# Real-time ticket classification
ticket = """User: Hi, I need help understanding my bill. The amount seems higher than usual.
Bot: I'll help you understand that. Could you provide your policy number?
User: Yes, it's 987654."""

prediction = classifier.predict(ticket)
print(f"Ticket category: {prediction[0][0]} (confidence: {prediction[0][1]:.2f})")

# Production evaluation pipeline
def evaluate_classifier(classifier, eval_data):
    """Evaluate classifier on insurance customer service examples"""
    correct = 0
    for item in eval_data:
        pred = classifier.predict(item['input'])
        if pred[0][0] == item['target']:
            correct += 1
    return correct / len(eval_data)

# Save trained classifier for production use
classifier.save("./insurance_classifier")
```

## Memory Relation
- **Previous**: [Query Safety Classification](./query_safety_classification.md)
- **Next**: None (Final memory)

## File Information
- **Path**: support_ticket_classification.py
- **Lines**: L1-278

## Tags
- insurance_automation
- customer_service
- ticket_classification
- intent_detection
- conversational_ai
- production_insurance
- support_automation
