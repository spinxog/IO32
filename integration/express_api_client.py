import requests
import os
from authlib.integrations.requests_client import OAuth2Session

AUTHORIZATION_URL = "https://ims.adobe.com/authorize/v2"
TOKEN_URL = "https://ims.adobe.com/token/v2"

class ExpressAPIClient:
    def __init__(self, client_id=None, client_secret=None, redirect_uri=None, token=None):
        self.client_id = client_id or os.getenv("ADOBE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ADOBE_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("ADOBE_REDIRECT_URI")
        self.session = OAuth2Session(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri
        )
        if token:
            self.session.token = token

    def get_auth_url(self, scope="openid profile"):
        uri, state = self.session.create_authorization_url(AUTHORIZATION_URL, scope=scope)
        return uri, state

    def fetch_token(self, authorization_response_url):
        token = self.session.fetch_token(TOKEN_URL, authorization_response=authorization_response_url)
        return token

    def get_project(self, project_id):
        resp = self.session.get(f"https://api.adobe.com/express/v1/projects/{project_id}")
        resp.raise_for_status()
        return resp.json()