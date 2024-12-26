"""
Documentation for the /generate-multiple-disease-definitions endpoint
"""
from flask_restx import Resource

def register_disease_endpoint(ns, models):
    """Register the disease endpoint with the given namespace"""

    @ns.route('/generate-multiple-disease-definitions')
    class GenerateMultipleDiseaseDefinitions(Resource):
        """Endpoint for generating disease definitions"""

        @ns.doc(description='Generate multiple disease definitions')
        @ns.expect(models['disease_request'])
        @ns.response(200, 'Success', models['disease_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Generate multiple disease definitions
            
            This endpoint generates structured definitions for multiple diseases,
            including symptoms, lab results, procedures, and risk factors.
            """
            pass  # Implementation is in app.py 