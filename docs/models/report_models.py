"""
Models for the /api/disease-scanner/reports/generate-report endpoint
"""
from flask_restx import fields

def init_report_models(api):
    """Initialize models for the report generation endpoint"""
    
    finding_model = api.model('Finding', {
        'text': fields.String(required=True, description='The matched text from the document'),
        'confidence': fields.Float(required=True, description='Confidence score of the match'),
        'page': fields.Integer(description='Page number where the finding was found'),
        'document_name': fields.String(description='Name of the document containing the finding'),
        'source_type': fields.String(description='Type of source document (e.g., medical_record, lab_report)')
    })

    disease_match_model = api.model('DiseaseMatch', {
        'disease_name': fields.String(required=True, description='Name of the disease'),
        'confidence_score': fields.Float(required=True, description='Overall confidence score for the disease'),
        'findings': fields.List(fields.Nested(finding_model), description='List of findings supporting the disease'),
        'category_scores': fields.Raw(description='Scores broken down by category (symptoms, lab results, etc.)')
    })

    report_request = api.model('ReportRequest', {
        'userId': fields.String(required=True, description='ID of the user requesting the report'),
        'projectId': fields.String(required=True, description='ID of the project'),
        'scanResults': fields.List(fields.Nested(disease_match_model), required=True, description='Results from disease scanning'),
        'format': fields.String(description='Desired format of the report (e.g., PDF, DOCX)', default='PDF')
    })

    report_response = api.model('ReportResponse', {
        'status': fields.String(required=True, description='Status of the report generation'),
        'message': fields.String(required=True, description='Detailed message about the operation'),
        'reportUrl': fields.String(description='URL to download the generated report'),
        'reportData': fields.Raw(description='Report data if requested in raw format')
    })

    return {
        'finding': finding_model,
        'disease_match': disease_match_model,
        'report_request': report_request,
        'report_response': report_response
    } 