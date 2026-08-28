from __future__ import annotations

from dataclasses import dataclass
import os
from contextlib import contextmanager

import boto3


@dataclass(frozen=True)
class AwsSessionContext:
    """Resolved AWS session and frozen credentials for local pipeline jobs."""

    session: boto3.Session
    profile_name: str | None
    access_key: str
    secret_key: str
    token: str | None


def _candidate_profiles() -> list[str | None]:
    """Return profile candidates in the order we should try them."""
    candidates: list[str | None] = []
    for env_name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        value = os.environ.get(env_name)
        if value and value not in candidates:
            candidates.append(value)
    return candidates


@contextmanager
def _temporarily_unset_env(*names: str):
    """Temporarily clear environment variables while probing fallback credentials."""
    snapshot = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in snapshot.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _frozen_credentials(session: boto3.Session):
    """Return frozen credentials if the session can resolve them."""
    credentials = session.get_credentials()
    if credentials is None:
        return None
    frozen = credentials.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        return None
    return frozen


def build_aws_session(region_name: str | None = None) -> AwsSessionContext:
    """Resolve a usable AWS session, falling back to the default profile if needed."""
    last_error: Exception | None = None
    for profile_name in _candidate_profiles():
        try:
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            frozen = _frozen_credentials(session)
            if frozen is None:
                continue
            return AwsSessionContext(
                session=session,
                profile_name=profile_name,
                access_key=frozen.access_key,
                secret_key=frozen.secret_key,
                token=frozen.token,
            )
        except Exception as exc:  # noqa: BLE001 - credential resolution failures vary by backend
            last_error = exc

    try:
        with _temporarily_unset_env("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
            session = boto3.Session(region_name=region_name)
            frozen = _frozen_credentials(session)
            if frozen is not None:
                return AwsSessionContext(
                    session=session,
                    profile_name=None,
                    access_key=frozen.access_key,
                    secret_key=frozen.secret_key,
                    token=frozen.token,
                )
    except Exception as exc:  # noqa: BLE001 - credential resolution failures vary by backend
        last_error = exc

    profile_hint = os.environ.get("AWS_PROFILE")
    default_hint = os.environ.get("AWS_DEFAULT_PROFILE")
    raise RuntimeError(
        "Unable to resolve AWS credentials. "
        f"Tried AWS_PROFILE={profile_hint!r}, AWS_DEFAULT_PROFILE={default_hint!r}, and default credentials."
    ) from last_error
