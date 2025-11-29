import bcrypt


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    :param password: Plain text password
    :return: Hashed password
    :raises ValueError: If password exceeds bcrypt's 72-byte limit
    """
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        raise ValueError("Password too long. Maximum 72 bytes allowed for bcrypt.")

    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against hash.
    :param plain_password: Plain text password
    :param hashed_password: Hashed password
    :return: True if password matches, else False.
    """
    password_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)
