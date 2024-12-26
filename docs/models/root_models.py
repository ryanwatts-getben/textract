"""
Models for the root endpoint
"""
from flask_restx import fields

def init_root_models(api):
    """Initialize models for the root endpoint"""
    
    redirect_response = api.model('RedirectResponse', {
        'location': fields.String(required=True, description='URL to redirect to', example='/swagger'),
        'status_code': fields.Integer(required=True, description='HTTP status code for redirect', example=302)
    })

    return {
        'redirect_response': redirect_response
    } 