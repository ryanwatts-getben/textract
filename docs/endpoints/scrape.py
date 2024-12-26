"""
Documentation for the /scrape endpoint
"""
from flask_restx import Resource

def register_scrape_endpoint(ns, models):
    """Register the scrape endpoint with the given namespace"""

    @ns.route('/')  # Changed to match actual route
    class Scrape(Resource):
        """Endpoint for scraping medical information"""

        @ns.doc(description='Extract disease information from medical sources')
        @ns.expect(models['scrape_request'])
        @ns.response(200, 'Success', models['scrape_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Extract disease information from medical sources
            
            This endpoint accepts disease names and extracts relevant medical information
            from trusted online sources.
            """
            pass  # Implementation is in app.py 