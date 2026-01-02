import {
    IExecuteFunctions,
    INodeExecutionData,
    INodeType,
    INodeTypeDescription,
    NodeOperationError,
} from 'n8n-workflow';

export class NodeAI implements INodeType {
    description: INodeTypeDescription = {
        displayName: 'Node-AI',
        name: 'nodeAI',
        icon: 'file:nodeai.svg',
        group: ['transform'],
        version: 1,
        subtitle: '={{$parameter["operation"]}}',
        description: 'Get intelligent AI-powered suggestions for your workflow',
        defaults: {
            name: 'Node-AI',
        },
        inputs: ['main'],
        outputs: ['main'],
        credentials: [
            {
                name: 'nodeAiApi',
                required: true,
            },
        ],
        properties: [
            // Operation Selection
            {
                displayName: 'Operation',
                name: 'operation',
                type: 'options',
                noDataExpression: true,
                options: [
                    {
                        name: 'Predict Next Node',
                        value: 'predict',
                        description: 'Get suggestions for the next node in your workflow',
                        action: 'Predict next node',
                    },
                    {
                        name: 'Analyze Workflow',
                        value: 'analyze',
                        description: 'Analyze current workflow for patterns and improvements',
                        action: 'Analyze workflow',
                    },
                    {
                        name: 'Generate Workflow',
                        value: 'generate',
                        description: 'Generate a workflow skeleton from a goal description',
                        action: 'Generate workflow',
                    },
                ],
                default: 'predict',
            },

            // Predict Operation Fields
            {
                displayName: 'Current Nodes',
                name: 'currentNodes',
                type: 'string',
                default: '',
                placeholder: 'webhook, set, http_request',
                description: 'Comma-separated list of node types in your current workflow',
                displayOptions: {
                    show: {
                        operation: ['predict'],
                    },
                },
            },
            {
                displayName: 'Number of Suggestions',
                name: 'topK',
                type: 'number',
                typeOptions: {
                    minValue: 1,
                    maxValue: 10,
                },
                default: 5,
                description: 'How many suggestions to return',
                displayOptions: {
                    show: {
                        operation: ['predict'],
                    },
                },
            },

            // Generate Operation Fields
            {
                displayName: 'Goal Description',
                name: 'goal',
                type: 'string',
                default: '',
                placeholder: 'Create a WhatsApp chatbot with AI responses',
                description: 'Natural language description of what you want to build',
                displayOptions: {
                    show: {
                        operation: ['generate'],
                    },
                },
            },
            {
                displayName: 'Starting Node',
                name: 'startNode',
                type: 'string',
                default: 'webhook',
                description: 'The type of node to start the workflow with',
                displayOptions: {
                    show: {
                        operation: ['generate'],
                    },
                },
            },
            {
                displayName: 'Maximum Nodes',
                name: 'maxNodes',
                type: 'number',
                typeOptions: {
                    minValue: 3,
                    maxValue: 20,
                },
                default: 10,
                description: 'Maximum number of nodes to generate',
                displayOptions: {
                    show: {
                        operation: ['generate'],
                    },
                },
            },

            // Analyze Operation Fields
            {
                displayName: 'Workflow JSON',
                name: 'workflowJson',
                type: 'json',
                default: '{}',
                description: 'The workflow JSON to analyze (usually from $workflow)',
                displayOptions: {
                    show: {
                        operation: ['analyze'],
                    },
                },
            },
        ],
    };

    async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
        const items = this.getInputData();
        const returnData: INodeExecutionData[] = [];

        const credentials = await this.getCredentials('nodeAiApi');
        const apiUrl = credentials.apiUrl as string;

        for (let i = 0; i < items.length; i++) {
            try {
                const operation = this.getNodeParameter('operation', i) as string;
                let responseData: any;

                if (operation === 'predict') {
                    const currentNodesStr = this.getNodeParameter('currentNodes', i) as string;
                    const topK = this.getNodeParameter('topK', i) as number;

                    const currentNodes = currentNodesStr
                        .split(',')
                        .map(n => n.trim().toLowerCase())
                        .filter(n => n.length > 0);

                    const response = await this.helpers.request({
                        method: 'POST',
                        url: `${apiUrl}/predict`,
                        body: {
                            current_nodes: currentNodes,
                            top_k: topK,
                        },
                        json: true,
                    });

                    responseData = response;

                } else if (operation === 'generate') {
                    const goal = this.getNodeParameter('goal', i) as string;
                    const startNode = this.getNodeParameter('startNode', i) as string;
                    const maxNodes = this.getNodeParameter('maxNodes', i) as number;

                    const response = await this.helpers.request({
                        method: 'POST',
                        url: `${apiUrl}/generate`,
                        body: {
                            goal: goal,
                            start_node: startNode,
                            max_nodes: maxNodes,
                        },
                        json: true,
                    });

                    responseData = response;

                } else if (operation === 'analyze') {
                    const workflowJson = this.getNodeParameter('workflowJson', i) as object;

                    const response = await this.helpers.request({
                        method: 'POST',
                        url: `${apiUrl}/analyze`,
                        body: {
                            workflow: workflowJson,
                            find_patterns: true,
                            suggest_improvements: true,
                        },
                        json: true,
                    });

                    responseData = response;
                }

                returnData.push({
                    json: responseData,
                    pairedItem: { item: i },
                });

            } catch (error) {
                if (this.continueOnFail()) {
                    returnData.push({
                        json: {
                            error: (error as Error).message,
                        },
                        pairedItem: { item: i },
                    });
                    continue;
                }
                throw new NodeOperationError(this.getNode(), error as Error, { itemIndex: i });
            }
        }

        return [returnData];
    }
}
