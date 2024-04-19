import requests
from urllib.parse import urljoin
from requests.exceptions import RequestException

class API:
    def __init__(self, uri):
        self.base_uri = uri
        self.session = requests.Session()
        self.api_key = None

    def authorize(self, api_key):
        self.api_key = f"apikey {api_key}"

    def _request(self, method, path, **kwargs):
        headers = kwargs.get('headers', {})
        if self.api_key:
            headers['Authorization'] = self.api_key
        kwargs['headers'] = headers

        url = urljoin(self.base_uri, path)

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.content
        except RequestException as e:
            raise Exception(f"Failed to execute {method} request to {url}") from e

    def get(self, path, **kwargs):
        return self._request('GET', path, **kwargs).decode('utf-8')

    def post(self, path, data=None, json=None, **kwargs):
        if json:
            kwargs['json'] = json
        else:
            kwargs['data'] = data
        return self._request('POST', path, **kwargs).decode('utf-8')

    def put(self, path, data=None, json=None, **kwargs):
        if json:
            kwargs['json'] = json
        else:
            kwargs['data'] = data
        return self._request('PUT', path, **kwargs).decode('utf-8')

    def delete(self, path, **kwargs):
        return self._request('DELETE', path, **kwargs).decode('utf-8')
