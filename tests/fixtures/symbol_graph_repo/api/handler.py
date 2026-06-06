"""Top-level request handler. The longest call chain in the fixture
starts here: handle_request -> fetch_user -> query_db -> User.__init__.
"""
from api.auth import validate_token
from db.connection import query_db
from db.models import User
from utils.log import log_info


def handle_request(req):
    token = req.get("token")
    if not validate_token(token):
        return {"error": "unauthorized"}
    user = fetch_user(req["user_id"])
    log_info(f"served {user}")
    return {"user": user.full_name()}


def fetch_user(user_id):
    row = query_db("SELECT * FROM users WHERE id = ?", [user_id])
    return User(row)
