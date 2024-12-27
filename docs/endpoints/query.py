"""
Documentation for the /query endpoint
"""
from flask_restx import Resource

def register_query_endpoint(ns, models):
    """Register the query endpoint with the given namespace"""

    @ns.route('/query')
    class Query(Resource):
        """Endpoint for querying indexed documents"""

        @ns.doc(description='Query indexed documents')
        @ns.expect(models['query_request'])
        @ns.response(200, 'Success', models['query_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Query indexed documents
            
            This endpoint allows querying of previously indexed documents using
            natural language questions and returns relevant context and answers.
            """
            pass  # Implementation is in app.py 