"""Models for the scan endpoints"""
from flask_restx import fields

def init_scan_models(api):
    """Initialize models for scan operations"""
    
    # Request Models
    scan_request = api.model('ScanRequest', {
        'userId': fields.String(required=True, description='Unique identifier for the user'),
        'projectId': fields.String(required=True, description='Unique identifier for the project'),
        'massTorts': fields.List(fields.Raw, required=True, description='List of mass torts with diseases to scan for')
    })

    # Response Models
    finding_match = api.model('FindingMatch', {
        'matched_text': fields.String(description='Text from document that matches the finding'),
        'confidence_score': fields.Float(description='Confidence score for this match'),
        'page_number': fields.Integer(description='Page number where match was found'),
        'document_name': fields.String(description='Name of document containing the match'),
        'source_type': fields.String(description='Type of source document (e.g. medical_record, incident_report)')
    })

    disease_match = api.model('DiseaseMatch', {
        'disease_name': fields.String(description='Name of the matched disease'),
        'confidence_score': fields.Float(description='Overall confidence score for disease match'),
        'findings': fields.List(fields.Nested(finding_match), description='List of matched findings'),
        'category_scores': fields.Raw(description='Confidence scores by category')
    })

    scan_response = api.model('ScanResponse', {
        'status': fields.String(description='Status of the scan operation'),
        'message': fields.String(description='Descriptive message about the operation'),
        'matches': fields.List(fields.Nested(disease_match), description='List of disease matches found'),
        'scoring_summary': fields.Raw(description='Summary of scoring results')
    })

    begin_scan_response = api.model('BeginScanResponse', {
        'status': fields.String(description='Status of the scan operation'),
        'message': fields.String(description='Descriptive message about the operation'),
        'results': fields.List(fields.Raw, description='List of scan results'),
        'processing_time': fields.Float(description='Time taken to process the scan'),
        'total_diseases': fields.Integer(description='Total number of diseases processed'),
        'successful_diseases': fields.Integer(description='Number of diseases successfully processed')
    })

    return {
        'scan_request': scan_request,
        'finding_match': finding_match,
        'disease_match': disease_match,
        'scan_response': scan_response,
        'beginScan_response': begin_scan_response
    } 