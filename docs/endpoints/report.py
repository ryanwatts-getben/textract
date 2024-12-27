"""
Documentation for the /api/disease-scanner/reports/generate-report endpoint
"""
from flask_restx import Resource

def register_report_endpoint(ns, models):
    """Register the report endpoint with the given namespace"""

    @ns.route('/api/disease-scanner/reports/generate-report')
    class GenerateReport(Resource):
        """Endpoint for generating disease scan reports"""

        @ns.doc(description='Generate disease scan report')
        @ns.expect(models['report_request'])
        @ns.response(200, 'Success', models['report_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Generate disease scan report
            
            This endpoint generates a comprehensive report based on the scan results,
            including disease matches, confidence scores, and supporting evidence.
            """
            pass  # Implementation is in app.py 