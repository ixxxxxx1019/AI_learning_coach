"""用户认证模块测试。"""

import os
import tempfile
import threading
from pathlib import Path

import pytest

from utils.auth import AuthManager


@pytest.fixture
def auth():
    """创建临时文件路径的 AuthManager。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    mgr = AuthManager.__new__(AuthManager)
    mgr._filepath = path
    mgr._users = {}
    mgr._tokens = {}
    mgr._lock = threading.Lock()  # new lock (不能直接用 _thread.lock)
    AuthManager._instance = mgr
    yield mgr
    os.unlink(path)
    AuthManager._instance = None


class TestRegister:
    """注册测试。"""

    def test_register_success(self, auth):
        user, token = auth.register("alice", "secret123")
        assert user.user_id == "alice"
        assert user.username == "alice"
        assert len(token) == 64

    def test_register_duplicate(self, auth):
        auth.register("bob", "password1")
        with pytest.raises(ValueError, match="已存在"):
            auth.register("bob", "password2")

    def test_register_short_username(self, auth):
        with pytest.raises(ValueError, match="3-32"):
            auth.register("ab", "123456")

    def test_register_short_password(self, auth):
        with pytest.raises(ValueError, match="至少 6 位"):
            auth.register("alice", "12345")

    def test_register_username_trim_and_lower(self, auth):
        user, _ = auth.register("  Alice  ", "123456")
        assert user.username == "alice"


class TestLogin:
    """登入测试。"""

    def test_login_success(self, auth):
        auth.register("alice", "secret123")
        token = auth.login("alice", "secret123")
        assert len(token) == 64

    def test_login_wrong_password(self, auth):
        auth.register("alice", "secret123")
        with pytest.raises(ValueError, match="用户名或密码错误"):
            auth.login("alice", "wrongpassword")

    def test_login_unknown_user(self, auth):
        with pytest.raises(ValueError, match="用户名或密码错误"):
            auth.login("nobody", "password")

    def test_login_updates_last_login(self, auth):
        _, token = auth.register("alice", "secret123")
        # 登入后 token 和 last_login 都应更新
        auth.logout(token)
        auth.login("alice", "secret123")
        user = auth.get_user("alice")
        # 验证 last_login 字段存在且非空
        assert user.last_login
        assert len(user.last_login) > 0


class TestToken:
    """Token 验证测试。"""

    def test_validate_token(self, auth):
        _, token = auth.register("alice", "secret123")
        user = auth.validate_token(token)
        assert user is not None
        assert user.user_id == "alice"

    def test_invalid_token(self, auth):
        user = auth.validate_token("invalid-token")
        assert user is None

    def test_logout_invalidates_token(self, auth):
        _, token = auth.register("alice", "secret123")
        auth.logout(token)
        user = auth.validate_token(token)
        assert user is None

    def test_new_login_invalidates_old_token(self, auth):
        auth.register("alice", "secret123")
        token1 = auth.login("alice", "secret123")
        token2 = auth.login("alice", "secret123")
        assert token1 != token2
        assert auth.validate_token(token1) is None  # old token invalid
        assert auth.validate_token(token2) is not None  # new token valid
