"""
Models for scrape operations
"""
from flask_restx import fields

def init_scrape_models(api):
    """Initialize models for scrape operations"""
    
    # Scrape Request Model
    scrape_request = api.model('ScrapeRequest', {
        'diseaseUrl': fields.Raw(required=True, 
            description='URL or array of [URL, name] pairs. Can be either a single URL string or an array of [URL, name] pairs.',
            example=[
                ["https://medlineplus.gov/genetics/condition/non-alcoholic-fatty-liver-disease/", 
                 "Non-Alcoholic Fatty Liver Disease"]
            ])
    })

    # Scrape Response Model
    scrape_response = api.model('ScrapeResponse', {
        'title': fields.String(description='Disease title', 
            example='Non-Alcoholic Fatty Liver Disease'),
        'also_called': fields.String(description='Alternative names for the disease',
            example='NAFLD, Fatty Liver Disease'),
        'summary': fields.String(description='Disease summary',
            example='Non-alcoholic fatty liver disease is a condition in which excess fat builds up in liver cells...'),
        'symptoms': fields.List(fields.String, description='List of symptoms',
            example=[
                'Fatigue',
                'Pain in upper right abdomen',
                'Enlarged liver',
                'Jaundice'
            ]),
        'lab_results': fields.List(fields.String, description='List of laboratory tests',
            example=[
                'Liver function tests',
                'Complete blood count',
                'Lipid profile'
            ]),
        'diagnostic_procedures': fields.List(fields.String, description='List of diagnostic procedures',
            example=[
                'Ultrasound',
                'CT scan',
                'Liver biopsy'
            ]),
        'treatments': fields.List(fields.String, description='List of treatments',
            example=[
                'Weight loss',
                'Dietary changes',
                'Exercise program',
                'Medications'
            ]),
        'risk_factors': fields.List(fields.String, description='List of risk factors',
            example=[
                'Obesity',
                'Type 2 diabetes',
                'High cholesterol'
            ]),
        'complications': fields.List(fields.String, description='List of potential complications',
            example=[
                'Cirrhosis',
                'Liver cancer',
                'Liver failure'
            ]),
        'prevention': fields.List(fields.String, description='List of prevention methods',
            example=[
                'Maintain healthy weight',
                'Exercise regularly',
                'Eat a balanced diet'
            ]),
        'when_to_see_doctor': fields.List(fields.String, description='When to seek medical attention',
            example=[
                'Severe abdominal pain',
                'Yellowing of skin or eyes',
                'Dark urine color'
            ])
    })

    # Multiple Response Model
    scrape_multiple_response = api.model('ScrapeMultipleResponse', {
        '*': fields.Nested(scrape_response, description='Response for each disease, keyed by disease name')
    })

    return {
        'scrape_request': scrape_request,
        'scrape_response': scrape_response,
        'scrape_multiple_response': scrape_multiple_response
    } 