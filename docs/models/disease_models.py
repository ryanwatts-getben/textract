"""
Models for the /generate-multiple-disease-definitions endpoint
"""
from flask_restx import fields

def init_disease_models(api):
    """Initialize models for the disease definitions endpoint"""
    
    disease_criteria = api.model('DiseaseDefinitionCriteria', {
        'name': fields.String(required=True, description='Name of the disease'),
        'symptoms': fields.List(fields.String, description='List of symptoms associated with the disease'),
        'lab_results': fields.List(fields.String, description='List of relevant lab results'),
        'procedures': fields.List(fields.String, description='List of diagnostic procedures'),
        'risk_factors': fields.List(fields.String, description='List of risk factors'),
        'treatments': fields.List(fields.String, description='List of treatments'),
        'complications': fields.List(fields.String, description='List of potential complications')
    })

    disease_request = api.model('DiseaseDefinitionsRequest', {
        'diseases': fields.List(fields.Nested(disease_criteria), required=True, description='List of diseases to generate definitions for')
    })

    disease_definition = api.model('DiseaseDefinition', {
        'name': fields.String(required=True, description='Name of the disease'),
        'definition': fields.String(required=True, description='Generated definition of the disease'),
        'criteria': fields.Nested(disease_criteria, description='Original criteria used for generation')
    })

    disease_response = api.model('DiseaseDefinitionsResponse', {
        'status': fields.String(required=True, description='Status of the generation operation'),
        'message': fields.String(required=True, description='Detailed message about the operation'),
        'definitions': fields.List(fields.Nested(disease_definition), required=True, description='List of generated disease definitions')
    })

    return {
        'disease_criteria': disease_criteria,
        'disease_request': disease_request,
        'disease_definition': disease_definition,
        'disease_response': disease_response
    } 