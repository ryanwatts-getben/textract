"""
Documentation for the /query endpoint
"""
from flask_restx import Resource

def register_query_endpoint(ns, models):
    """Register the query endpoint with the given namespace"""

    @ns.route('/query')
    class Query(Resource):
        """Endpoint for querying indexed documents"""

        @ns.doc(description='Query the indexed documents using natural language')
        @ns.expect(models['query_request'])
        @ns.response(200, 'Success', models['query_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Query indexed documents
            
            This endpoint accepts a natural language query and returns relevant information
            from the previously indexed documents. The response includes both the answer
            and the relevant context from the documents.
            
            Example request:
            {
                "userId": "user123",
                "projectId": "project456",
                "query": "What medications were prescribed for the patient's condition?"
            }
            """
            pass  # Implementation is in app.py 