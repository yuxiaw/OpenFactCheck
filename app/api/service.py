import json

from cmn.api import API

class FactCheckAPI(API):
    def __init__(self, uri):
        super().__init__(uri)

    def health(self):
        return self.get('/health')

    def list_solvers(self):
        return json.loads(self.get('/pipeline/solver'))
    
    def list_claim_processors(self):
        return json.loads(self.get('/pipeline/solver/claimprocessor'))
    
    def list_retrievers(self):
        return json.loads(self.get('/pipeline/solver/retriever'))
    
    def list_verifiers(self):
        return json.loads(self.get('/pipeline/solver/verifier'))
    
    def pipeline_configuration(self):
        return json.loads(self.get('/pipeline/configuration'))
    
    def pipeline_configure(self, claim_processor: dict, retriever: dict, verifier: dict):
        return json.loads(self.post('/pipeline/configure', json={**claim_processor, **retriever, **verifier}))

    def evaluate_response(self, text):
        return json.loads(self.post('/evaluate/response', json={'text': text}))
    
    def evaluate_llm(self, id):
        return json.loads(self.post(f'/evaluate/llm/{id}'))
    
    def evaluate_factchecker(self, id):
        return json.loads(self.post(f'/evaluate/factchecker/{id}'))
    
