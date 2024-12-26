"""
Models for scan operations
"""
from flask_restx import fields

def init_scan_models(api):
    """Initialize models for scan operations"""
    
    # Base models for request
    lab_range_model = api.model('LabRange', {
        'name': fields.String(description='Lab test name with normal range',
            example='Complete Blood Count: 4.5-11.0 x10^9/L'),
        'range': fields.String(description='Additional range information',
            example='Not specified')
    })

    scoring_model = api.model('ScoringModel', {
        'id': fields.String(description='Scoring model ID',
            example='e4d52e91-1114-4264-9dcf-71bec597c0b3'),
        'confidenceThreshold': fields.Float(description='Confidence threshold for matches',
            example=0.7)
    })

    disease_model = api.model('Disease', {
        'id': fields.String(description='Disease ID',
            example='57bab3b6-41c6-4255-bcb0-e0fb242a12b9'),
        'name': fields.String(description='Disease name',
            example='Kidney Cancer'),
        'symptoms': fields.List(fields.String, description='List of symptoms',
            example=[
                'Blood in urine',
                'Lower back pain',
                'Fatigue',
                'Weight loss'
            ]),
        'labResults': fields.List(fields.Nested(lab_range_model),
            description='List of lab results with ranges'),
        'diagnosticProcedures': fields.List(fields.String,
            description='List of diagnostic procedures',
            example=[
                'CT scan',
                'MRI',
                'Ultrasound',
                'Biopsy'
            ]),
        'riskFactors': fields.List(fields.String,
            description='List of risk factors',
            example=[
                'Smoking',
                'Obesity',
                'Hypertension',
                'Family history'
            ]),
        'scoringModel': fields.Nested(scoring_model)
    })

    mass_tort_model = api.model('MassTort', {
        'id': fields.String(description='Mass tort ID',
            example='02ccfcb0-1658-4242-a4c9-0b0d4de17e85'),
        'officialName': fields.String(description='Official name of the mass tort',
            example='Aqueous Film-Forming Foam'),
        'diseases': fields.List(fields.Nested(disease_model))
    })

    # Scan Request Model
    scan_request = api.model('ScanRequest', {
        'userId': fields.String(required=True,
            description='User ID',
            example='4e6f9de5-5128-4ff4-980c-2429e31ec2ec'),
        'projectId': fields.String(required=True,
            description='Project ID',
            example='ea2fc6b9-7759-470f-9e48-93b364eee330'),
        'massTorts': fields.List(fields.Nested(mass_tort_model), required=True,
            description='List of mass torts with their associated diseases')
    })

    # Response Models
    match_model = api.model('Match', {
        'name': fields.String(description='Name of the matched item'),
        'confidence': fields.Float(description='Confidence score of the match (0-1)',
            example=0.85),
        'excerpt': fields.String(description='Relevant text excerpt containing the match',
            example='Patient exhibits severe headaches with associated nausea and sensitivity to light')
    })

    category_scores_model = api.model('CategoryScores', {
        'symptoms': fields.Float(description='Confidence score for symptoms category (0-1)',
            example=0.75),
        'lab_results': fields.Float(description='Confidence score for lab results category (0-1)',
            example=0.85),
        'procedures': fields.Float(description='Confidence score for procedures category (0-1)',
            example=0.65),
        'risk_factors': fields.Float(description='Confidence score for risk factors category (0-1)',
            example=0.70)
    })

    disease_analysis_model = api.model('DiseaseAnalysis', {
        'status': fields.String(description='Status of the disease analysis',
            enum=['success', 'no_matches', 'error'],
            example='success'),
        'disease_name': fields.String(description='Name of the analyzed disease',
            example='Traumatic Brain Injury'),
        'matches': fields.Nested(api.model('Matches', {
            'symptoms': fields.List(fields.Nested(match_model)),
            'lab_results': fields.List(fields.Nested(match_model)),
            'procedures': fields.List(fields.Nested(match_model)),
            'risk_factors': fields.List(fields.Nested(match_model)),
            'confidence_scores': fields.Nested(category_scores_model),
            'overall_confidence': fields.Float(description='Overall confidence score for the disease (0-1)',
                example=0.78),
            'relevant_excerpts': fields.List(fields.String,
                description='List of relevant text excerpts with confidence context')
        }))
    })

    mass_tort_analysis_model = api.model('MassTortAnalysis', {
        'mass_tort_name': fields.String(description='Name of the mass tort',
            example='Aqueous Film-Forming Foam'),
        'diseases': fields.List(fields.Nested(disease_analysis_model)),
        'processing_time': fields.Float(description='Time taken to process this mass tort in seconds',
            example=2.45)
    })

    beginScan_response = api.model('BeginScanResponse', {
        'status': fields.String(required=True,
            description='Overall status of the scan operation',
            enum=['success', 'error'],
            example='success'),
        'message': fields.String(description='Status message or error description',
            example='Successfully processed 3 diseases'),
        'results': fields.List(fields.Nested(mass_tort_analysis_model)),
        'processing_time': fields.Float(description='Total processing time in seconds',
            example=5.67),
        'total_diseases': fields.Integer(description='Total number of diseases analyzed',
            example=3),
        'successful_diseases': fields.Integer(description='Number of diseases successfully analyzed',
            example=3)
    })

    return {
        'scan_request': scan_request,
        'match': match_model,
        'category_scores': category_scores_model,
        'disease_analysis': disease_analysis_model,
        'mass_tort_analysis': mass_tort_analysis_model,
        'beginScan_response': beginScan_response
    } 