"""
Pydantic Schemas for Node-AI API
=================================
Request/Response models for the prediction service.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==================== Request Models ====================

class PredictRequest(BaseModel):
    """Request for next-node prediction"""
    current_nodes: List[str] = Field(
        ...,
        description="List of current node types in the workflow",
        example=["webhook", "set_data", "http_request"]
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top predictions to return"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "current_nodes": ["webhook", "set_data"],
                "top_k": 5
            }
        }


class AnalyzeRequest(BaseModel):
    """Request for workflow analysis and MacroNode discovery"""
    workflow: Dict[str, Any] = Field(
        ...,
        description="Workflow JSON in n8n format"
    )
    find_patterns: bool = Field(
        default=True,
        description="Whether to find MacroNode patterns"
    )
    suggest_improvements: bool = Field(
        default=True,
        description="Whether to suggest node replacements"
    )


class GenerateRequest(BaseModel):
    """Request for workflow skeleton generation"""
    goal: str = Field(
        ...,
        description="Natural language description of the workflow goal",
        example="Create a WhatsApp chatbot with AI responses"
    )
    start_node: Optional[str] = Field(
        default=None,
        description="Optional starting node type",
        example="webhook"
    )
    max_nodes: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Maximum number of nodes to generate"
    )


# ==================== Response Models ====================

class NodePrediction(BaseModel):
    """A single node prediction with confidence"""
    node_type: str = Field(..., description="Predicted node type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    usage_hint: Optional[str] = Field(None, description="Usage suggestion")


class PredictResponse(BaseModel):
    """Response for next-node prediction"""
    predictions: List[NodePrediction] = Field(
        ...,
        description="Top-k node predictions"
    )
    context_used: List[str] = Field(
        ...,
        description="Node context used for prediction"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Model version used"
    )


class MacroNodePattern(BaseModel):
    """A discovered MacroNode pattern"""
    name: str = Field(..., description="Suggested MacroNode name")
    nodes: List[str] = Field(..., description="Node types in the pattern")
    frequency: int = Field(..., description="How often this pattern appears")
    suggested_action: str = Field(..., description="What to do with this pattern")


class ImprovementSuggestion(BaseModel):
    """A workflow improvement suggestion"""
    position: int = Field(..., description="Position in workflow")
    current_node: str = Field(..., description="Current node at position")
    suggested_node: str = Field(..., description="Suggested replacement")
    confidence: float = Field(..., description="Confidence in suggestion")
    reason: str = Field(..., description="Why this improvement is suggested")


class AnalyzeResponse(BaseModel):
    """Response for workflow analysis"""
    workflow_id: str = Field(..., description="Workflow identifier")
    node_count: int = Field(..., description="Number of nodes")
    edge_count: int = Field(..., description="Number of connections")
    patterns_found: List[MacroNodePattern] = Field(
        default=[],
        description="MacroNode patterns found"
    )
    improvements: List[ImprovementSuggestion] = Field(
        default=[],
        description="Suggested improvements"
    )
    optimization_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall optimization score"
    )


class GeneratedNode(BaseModel):
    """A node in the generated workflow"""
    node_type: str = Field(..., description="Node type")
    position: int = Field(..., description="Position in workflow")
    suggested_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Suggested configuration"
    )


class GenerateResponse(BaseModel):
    """Response for workflow generation"""
    goal: str = Field(..., description="Original goal")
    nodes: List[GeneratedNode] = Field(..., description="Generated node sequence")
    edges: List[Dict[str, int]] = Field(..., description="Connections between nodes")
    confidence: float = Field(..., description="Overall confidence in generation")
    explanation: str = Field(..., description="Why these nodes were chosen")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    vocab_size: int = Field(..., description="Number of known node types")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
