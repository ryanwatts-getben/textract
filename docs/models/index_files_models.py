"""
Models for the document indexing endpoint
"""
from flask_restx import fields

def init_index_files_models(api):
    """Initialize models for the document indexing endpoint"""
    
    document_model = api.model('Document', {
        'id': fields.String(required=True, description='Unique identifier for the document'),
        'name': fields.String(required=True, description='Name of the document'),
        'content': fields.String(required=True, description='Text content of the document'),
        'type': fields.String(required=True, description='Type of document (e.g., medical_record, incident_report)'),
        'metadata': fields.Raw(description='Additional metadata about the document')
    })

    index_request = api.model('IndexRequest', {
        'userId': fields.String(required=True, description='ID of the user making the request'),
        'projectId': fields.String(required=True, description='ID of the project'),
        'documents': fields.List(fields.Nested(document_model), required=True, description='List of documents to index')
    })

    index_response = api.model('IndexResponse', {
        'status': fields.String(required=True, description='Status of the indexing operation'),
        'message': fields.String(required=True, description='Detailed message about the operation'),
        'indexedDocuments': fields.Integer(required=True, description='Number of documents successfully indexed'),
        'errors': fields.List(fields.String, description='List of any errors encountered during indexing')
    })

    return {
        'document': document_model,
        'index_request': index_request,
        'index_response': index_response
    } 