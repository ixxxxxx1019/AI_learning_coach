"""
用户认证模块 —— 注册 / 登入 / Token 验证。

轻量级实现：JSON 文件存储 + bcrypt 密码哈希 + SHA256 Token。
无需外部数据库依赖，适合单机/小规模部署。

Usage:
    from utils.auth import AuthManager

    auth = AuthManager()
    user = auth.register("alice", "securepassword")
    token = auth.login("alice", "securepassword")
    user = auth.validate_token(token)
"""

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import bcrypt as _bcrypt

from config.logging_config import get_logger

logger = get_logger(__name__)

# ---- 路径常量 ----
_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "users.json"


@dataclass
class User:
    """用户数据模型。"""

    user_id: str
    username: str
    password_hash: str
    created_at: str = ""
    last_login: str = ""
    token: str = ""
    token_created: str = ""


class AuthManager:
    """用户认证管理器。

    单例模式：整个应用共享一个实例。
    """

    _instance: Optional["AuthManager"] = None
    _lock = threading.Lock()

    def __new__(cls, filepath: Path | str | None = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    fp = filepath or _DEFAULT_PATH
                    obj._filepath = Path(fp) if not isinstance(fp, Path) else fp
                    obj._users: dict[str, User] = {}
                    obj._tokens: dict[str, User] = {}
                    obj._lock = threading.Lock()
                    obj._load()
                    cls._instance = obj
        return cls._instance

    # ---- 公开 API ----

    def register(self, username: str, password: str) -> tuple[User, str]:
        """注册新用户。

        Args:
            username: 用户名（3-32 字符，字母数字下划线）
            password: 密码（≥6 字符）

        Returns:
            (User, token)

        Raises:
            ValueError: 用户名已存在或格式不合法
        """
        username = username.strip().lower()

        if len(username) < 3 or len(username) > 32:
            raise ValueError("用户名长度需在 3-32 字符之间")
        if not username.replace("_", "").isalnum():
            raise ValueError("用户名只能包含字母、数字和下划线")
        if len(password) < 6:
            raise ValueError("密码长度至少 6 位")

        with self._lock:
            if username in self._users:
                raise ValueError(f"用户名 '{username}' 已存在")

            user_id = username  # user_id = username（简化设计）
            user = User(
                user_id=user_id,
                username=username,
                password_hash=_bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode(),
                created_at=datetime.now().isoformat(),
            )
            token = self._generate_token(user)
            user.token = token
            user.token_created = datetime.now().isoformat()
            user.last_login = datetime.now().isoformat()

            self._users[user_id] = user
            self._tokens[token] = user
            self._save()

            logger.info("user_registered", user_id=user_id)
            return user, token

    def login(self, username: str, password: str) -> str:
        """登入 → 返回 token。

        Raises:
            ValueError: 用户名或密码错误
        """
        username = username.strip().lower()

        with self._lock:
            user = self._users.get(username)
            if not user:
                raise ValueError("用户名或密码错误")

            if not _bcrypt.checkpw(password.encode(), user.password_hash.encode()):
                raise ValueError("用户名或密码错误")

            # 生成新 token
            token = self._generate_token(user)
            # 删除旧 token
            if user.token and user.token in self._tokens:
                del self._tokens[user.token]

            user.token = token
            user.token_created = datetime.now().isoformat()
            user.last_login = datetime.now().isoformat()
            self._tokens[token] = user
            self._save()

            logger.info("user_login", user_id=user.user_id)
            return token

    def validate_token(self, token: str) -> User | None:
        """验证 token → 返回 User 或 None。"""
        with self._lock:
            return self._tokens.get(token)

    def logout(self, token: str) -> None:
        """登出 → 使 token 失效。"""
        with self._lock:
            user = self._tokens.pop(token, None)
            if user:
                user.token = ""
                user.token_created = ""
                self._save()
                logger.info("user_logout", user_id=user.user_id)

    def get_user(self, user_id: str) -> User | None:
        """按 user_id 查询用户。"""
        with self._lock:
            return self._users.get(user_id)

    # ---- 内部方法 ----

    def _generate_token(self, user: User) -> str:
        """生成随机 token（SHA256 + secrets）。"""
        raw = f"{user.user_id}:{secrets.token_hex(32)}:{datetime.now().isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load(self):
        """从 JSON 文件加载用户数据。"""
        if not self._filepath.exists():
            return
        try:
            with self._filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for uid, udata in data.get("users", {}).items():
                user = User(**udata)
                self._users[uid] = user
                if user.token:
                    self._tokens[user.token] = user
            logger.info("users_loaded", count=len(self._users))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("users_load_error", error=str(e))

    def _save(self):
        """保存用户数据到 JSON 文件。"""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "users": {
                uid: {
                    "user_id": u.user_id,
                    "username": u.username,
                    "password_hash": u.password_hash,
                    "created_at": u.created_at,
                    "last_login": u.last_login,
                    "token": u.token,
                    "token_created": u.token_created,
                }
                for uid, u in self._users.items()
            }
        }
        with self._filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
