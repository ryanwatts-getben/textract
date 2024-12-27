"""
Documentation for the /index_files endpoint
"""
from flask_restx import Resource

def register_index_files_endpoint(ns, models):
    """Register the document indexing endpoint with the given namespace"""

    @ns.route('/')
    class IndexFiles(Resource):
        """Endpoint for indexing documents"""

        @ns.doc(description='Index a set of documents for later analysis')
        @ns.expect(models['index_request'])
        @ns.response(200, 'Success', models['index_response'])
        @ns.response(400, 'Invalid request')
        @ns.response(500, 'Internal server error')
        def post(self):
            """
            Index a set of documents
            
            This endpoint accepts a list of documents and creates a searchable index for later analysis.
            The documents are processed and stored in a vector database for efficient semantic search.
            
            The endpoint will be available at: http://localhost:5001/index_files
            
            Example request:
            {
                "userId": "user123",
                "projectId": "project456",
                "documents": [
                    {
                        "id": "doc1",
                        "name": "medical_record_2023.pdf",
                        "content": "Patient presents with...",
                        "type": "medical_record",
                        "metadata": {
                            "date": "2023-12-26",
                            "provider": "Dr. Smith"
                        }
                    }
                ]
            }
            """
            pass  # Implementation is in app.py 