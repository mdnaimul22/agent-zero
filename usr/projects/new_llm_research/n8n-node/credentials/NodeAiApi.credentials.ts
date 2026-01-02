import {
    ICredentialType,
    INodeProperties,
} from 'n8n-workflow';

export class NodeAiApi implements ICredentialType {
    name = 'nodeAiApi';
    displayName = 'Node-AI API';
    documentationUrl = 'https://github.com/your-repo/node-ai';

    properties: INodeProperties[] = [
        {
            displayName: 'API URL',
            name: 'apiUrl',
            type: 'string',
            default: 'http://localhost:8000',
            placeholder: 'http://localhost:8000',
            description: 'URL of the Node-AI prediction service',
        },
        {
            displayName: 'API Key',
            name: 'apiKey',
            type: 'string',
            typeOptions: {
                password: true,
            },
            default: '',
            description: 'Optional API key for authentication',
        },
    ];
}
