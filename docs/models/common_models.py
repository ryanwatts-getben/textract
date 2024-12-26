"""
Common models shared across endpoints
"""
from flask_restx import fields

def init_common_models(api):
    """Initialize common models shared across endpoints"""
    
    # Error Response Model
    error_response = api.model('ErrorResponse', {
        'error': fields.String(description='Error message',
            example='Invalid request format')
    })

    return {
        'error_response': error_response
    } 