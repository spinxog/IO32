import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integration.express_api_client import ExpressAPIClient

EXPORT_FOLDER = os.getenv("EXPORT_FOLDER", "./express_exported")

class ProjectRetriever:
    def __init__(self, api_client: ExpressAPIClient):
        self.client = api_client
        os.makedirs(EXPORT_FOLDER, exist_ok=True)

    def ensure_authenticated(self, auth_response_url=None):
        if not getattr(self.client.session, 'token', None):
            auth_url, state = self.client.get_auth_url(scope="openid profile https://img.adobe.io/s/ent-fonts")
            print("Visit this URL to authenticate:", auth_url)
            if auth_response_url:
                token = self.client.fetch_token(auth_response_url)
                self.client.session.token = token

    def fetch_project_assets(self, project_id: str, save_latest_only=True):
        project = self.client.get_project(project_id)
        assets = project.get("assets", [])
        saved_files = []
        for asset in assets:
            if asset.get("type") == "image" and asset.get("url"):
                url = asset["url"]
                filename = os.path.basename(url.split("?")[0])
                filepath = os.path.join(EXPORT_FOLDER, filename)
                with self.client.session.get(url, stream=True) as resp:
                    resp.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)
                saved_files.append(filepath)
        if save_latest_only and saved_files:
            to_delete = [f for f in os.listdir(EXPORT_FOLDER) if os.path.join(EXPORT_FOLDER, f) not in saved_files]
            for f in to_delete:
                os.remove(os.path.join(EXPORT_FOLDER, f))
        return saved_files
