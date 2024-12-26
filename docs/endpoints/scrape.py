"""
Documentation for the /scrape endpoint
"""
from flask_restx import Resource

def register_scrape_endpoint(scrape_ns, models):
    """Register the scrape endpoint with its documentation"""
    
    @scrape_ns.route('/scrape')
    class Scrape(Resource):
        """Endpoint for scraping disease information from medical sources."""
        
        @scrape_ns.doc('scrape_disease',
            description='Extract structured disease information from medical sources.')
        @scrape_ns.expect(models['scrape_request'], validate=True)
        @scrape_ns.response(200, 'Success', models['scrape_response'])
        @scrape_ns.response(400, 'Validation Error', models['error_response'])
        @scrape_ns.response(500, 'Internal Server Error', models['error_response'])
        def post(self):
            """
            Extract disease information from medical sources.
            
            This endpoint scrapes medical websites to extract structured information about diseases.
            It processes the content to identify and categorize various aspects of the disease.
            
            The extracted information includes:
            - Disease title and alternative names
            - Summary description
            - Symptoms and their characteristics
            - Laboratory tests and diagnostic procedures
            - Treatment options
            - Risk factors and complications
            - Prevention methods
            - Guidelines for seeking medical attention
            
            Example Usage:
            ```python
            {
                "diseaseUrl": [
                    ["https://medlineplus.gov/genetics/condition/non-alcoholic-fatty-liver-disease/",
                     "Non-Alcoholic Fatty Liver Disease"]
                ]
            }
            ```
            """
            pass  # Implementation is in scrape.py 