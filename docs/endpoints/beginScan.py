"""
Documentation for the /beginScan endpoint
"""
from flask_restx import Resource

def register_beginScan_endpoint(ns, models):
    """Register the beginScan endpoint with the given namespace"""

    @ns.route('/beginScan')
    class BeginScan(Resource):
        """Endpoint for scanning medical documents"""

        @ns.doc(description='Scan medical documents for disease evidence')
        @ns.expect(models['scan_request'])
        @ns.response(200, 'Success', models['scan_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Scan medical documents for disease evidence
            
            This endpoint analyzes medical documents to identify evidence of diseases
            and their characteristics, calculating confidence scores for matches.
            """
            pass  # Implementation is in app.py
