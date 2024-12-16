from typing import Any, Dict, List, Optional
from flask_restx import Api, fields  # type: ignore

# Initialize API
api: Api = Api(
    title='MedChron API',
    version='1.0',
    description='Medical Document Processing and Disease Information API',
    doc='/docs',
    ordered=True,
    validate=True
)

# Create namespaces
scrape_ns = api.namespace('scrape', description='Web scraping operations')

# Error Response Model
error_response = api.model('ErrorResponse', {
    'error': fields.String(description='Error message')
})

# Scrape Request/Response Models
scrape_request = api.model('ScrapeRequest', {
    'diseaseUrl': fields.Raw(required=True, 
        description='URL or array of [URL, name] pairs. Can be either a single URL string or an array of [URL, name] pairs.',
        example=[
            ["https://medlineplus.gov/genetics/condition/non-alcoholic-fatty-liver-disease/", 
             "Non-Alcoholic Fatty Liver Disease"]
        ])
})

scrape_response = api.model('ScrapeResponse', {
    'title': fields.String(description='Disease title', 
        example='Non-Alcoholic Fatty Liver Disease'),
    'also_called': fields.String(description='Alternative names for the disease',
        example='NAFLD, Fatty Liver Disease'),
    'summary': fields.String(description='Disease summary',
        example='Non-alcoholic fatty liver disease is a condition in which excess fat builds up in liver cells...'),
    'sections': fields.Raw(description='Various sections of the disease information',
        example={
            "Symptoms": "• Fatigue\n• Pain in upper right abdomen\n• Enlarged liver",
            "Causes": "• Obesity\n• Insulin resistance\n• High blood sugar",
            "Diagnosis and Tests": "• Blood tests\n• Imaging tests\n• Liver biopsy",
            "Treatments and Therapies": "• Weight loss\n• Healthy diet\n• Exercise",
            "Prevention": "• Maintain healthy weight\n• Exercise regularly\n• Eat a balanced diet"
        })
})

scrape_multiple_response = api.model('ScrapeMultipleResponse', {
    'Non-Alcoholic Fatty Liver Disease': fields.Nested(scrape_response),
    'Type 2 Diabetes': fields.Nested(scrape_response)
}) 