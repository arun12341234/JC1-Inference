"""
security.py - Security & Access Control Module
-----------------------------------------------
🔹 Features:
- JWT-based authentication for API access
- Password hashing & verification using bcrypt
- Role-based access control (RBAC)
- AES encryption for sensitive data

📌 Dependencies:
- jwt (for JSON Web Token authentication)
- bcrypt (for password hashing)
- cryptography (for AES encryption)
- datetime (for token expiry handling)
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

# Load security settings
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
TOKEN_EXPIRY_MINUTES = int(os.getenv("TOKEN_EXPIRY", 60))
AES_KEY = os.getenv("AES_SECRET_KEY", Fernet.generate_key())

# Initialize AES cipher
cipher = Fernet(AES_KEY)


### 🛡️ PASSWORD HASHING ###
def hash_password(password: str) -> str:
    """
    Hashes a password using bcrypt.
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verifies a password against a stored hash.
    """
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


### 🔑 JWT AUTHENTICATION ###
def create_jwt_token(user_id: str, role: str) -> str:
    """
    Generates a JWT token for authentication.
    
    📌 Payload:
    - user_id: Unique user identifier
    - role: User role (e.g., 'admin', 'user')
    - exp: Expiry timestamp
    """
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_jwt_token(token: str) -> dict:
    """
    Decodes and verifies a JWT token.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return {"error": "Token expired"}
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}


### 🔒 ROLE-BASED ACCESS CONTROL (RBAC) ###
ROLES_PERMISSIONS = {
    "admin": ["read", "write", "delete"],
    "user": ["read"],
    "editor": ["read", "write"],
}


def check_permission(role: str, action: str) -> bool:
    """
    Checks if a user role has permission for an action.
    """
    return action in ROLES_PERMISSIONS.get(role, [])


### 🔐 AES ENCRYPTION ###
def encrypt_data(data: str) -> str:
    """
    Encrypts data using AES encryption.
    """
    return cipher.encrypt(data.encode()).decode()


def decrypt_data(encrypted_data: str) -> str:
    """
    Decrypts AES-encrypted data.
    """
    return cipher.decrypt(encrypted_data.encode()).decode()


### 🛠️ Example Usage ###
if __name__ == "__main__":
    test_password = "SecurePass123"
    hashed_pass = hash_password(test_password)
    print(f"Password Hash: {hashed_pass}")

    is_valid = verify_password(test_password, hashed_pass)
    print(f"Password Verified: {is_valid}")

    token = create_jwt_token("user123", "admin")
    print(f"Generated JWT: {token}")

    decoded = verify_jwt_token(token)
    print(f"Decoded Token: {decoded}")

    encrypted = encrypt_data("Sensitive Information")
    print(f"Encrypted Data: {encrypted}")

    decrypted = decrypt_data(encrypted)
    print(f"Decrypted Data: {decrypted}")
