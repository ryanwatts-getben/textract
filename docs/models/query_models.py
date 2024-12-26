"""
Models for the /query endpoint
"""
from flask_restx import fields

def init_query_models(api):
    """Initialize models for the query endpoint"""
    
    query_request = api.model('QueryRequest', {
        'userId': fields.String(required=True, description='ID of the user making the request'),
        'projectId': fields.String(required=True, description='ID of the project'),
        'query': fields.String(required=True, description='The query text to search for in the indexed documents')
    })

    query_response = api.model('QueryResponse', {
        'status': fields.String(required=True, description='Status of the query operation'),
        'message': fields.String(required=True, description='Detailed message about the operation'),
        'response': fields.String(required=True, description='The response text from the query'),
        'context': fields.List(fields.String, description='Relevant document snippets that informed the response')
    })

    return {
        'query_request': query_request,
        'query_response': query_response
    } 