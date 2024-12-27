"""Models for the index endpoint"""
from flask_restx import fields

def init_index_models(api):
    """Initialize models for index operations"""
    
    # Request Models
    index_request = api.model('IndexRequest', {
        'userId': fields.String(required=True, description='Unique identifier for the user'),
        'projectId': fields.String(required=True, description='Unique identifier for the project'),
        'documents': fields.List(fields.Raw, required=True, description='List of documents to index')
    })

    # Response Models
    index_response = api.model('IndexResponse', {
        'status': fields.String(description='Status of the indexing operation'),
        'message': fields.String(description='Descriptive message about the operation'),
        'indexed_documents': fields.Integer(description='Number of documents successfully indexed'),
        'index_id': fields.String(description='Unique identifier for the created index')
    })

    return {
        'index_request': index_request,
        'index_response': index_response
    } 