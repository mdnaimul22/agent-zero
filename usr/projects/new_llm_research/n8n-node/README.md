# n8n-nodes-nodeai

🧠 **n8n Community Node for Node-AI Prediction Service**

Get intelligent AI-powered suggestions for your workflow directly in the n8n canvas.

## Installation

### Local Development

```bash
cd n8n-node
npm install
npm run build
npm link

# In your n8n installation
npm link n8n-nodes-nodeai
```

### Production

1. Start the Node-AI API server:
   ```bash
   cd /path/to/new_llm_research
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

2. In n8n, go to **Settings → Community Nodes** and install this package.

## Operations

### 1. Predict Next Node
Get AI-powered suggestions for what node should come next in your workflow.

**Input:**
- Current Nodes: Comma-separated list of node types (e.g., "webhook, set, http_request")
- Number of Suggestions: How many suggestions to return (1-10)

**Output:**
```json
{
  "predictions": [
    {"node_type": "condition", "confidence": 0.45},
    {"node_type": "code", "confidence": 0.30}
  ]
}
```

### 2. Analyze Workflow
Discover MacroNode patterns and get optimization suggestions.

**Input:**
- Workflow JSON: The current workflow (use `$workflow` expression)

**Output:**
```json
{
  "patterns_found": [...],
  "improvements": [...],
  "optimization_score": 0.85
}
```

### 3. Generate Workflow
Create a workflow skeleton from a natural language goal.

**Input:**
- Goal Description: e.g., "WhatsApp chatbot with AI"
- Starting Node: e.g., "webhook"
- Maximum Nodes: 3-20

**Output:**
```json
{
  "nodes": [
    {"node_type": "webhook", "position": 0},
    {"node_type": "agent", "position": 1}
  ],
  "edges": [...]
}
```

## Configuration

Add credentials for the Node-AI API:
- **API URL**: URL of your Node-AI server (default: `http://localhost:8000`)
- **API Key**: Optional authentication key

## License

MIT
