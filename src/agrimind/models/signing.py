"""Chữ ký số cho gói phát hành mô hình (Core → từng nước, A07 luồng 2).

Core ký bằng khoá bí mật Ed25519; điểm trong nước chỉ cài gói khi chữ ký đúng
với khoá công khai đã cài sẵn VÀ mã băm tệp mô hình khớp. Kẻ chen giữa đường
không đổi được mô hình hay nội dung thẻ mô hình.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .registry import ModelCard, Status, canonical, file_sha256, gate_problems


class BundleError(ValueError):
    pass


def new_keypair() -> tuple[bytes, bytes]:
    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return priv, pub


def _load_priv(pem: bytes) -> Ed25519PrivateKey:
    k = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(k, Ed25519PrivateKey):
        raise BundleError("khoá ký phải là Ed25519")
    return k


def _load_pub(pem: bytes) -> Ed25519PublicKey:
    k = serialization.load_pem_public_key(pem)
    if not isinstance(k, Ed25519PublicKey):
        raise BundleError("khoá công khai phải là Ed25519")
    return k


def sign_bundle(card: ModelCard, target: str, private_pem: bytes, issuer: str = "agrimind-core") -> dict:
    if card.status not in {Status.APPROVED, Status.PUBLISHED} or gate_problems(card):
        raise BundleError(f"{card.key} chưa qua cửa duyệt, không ký phát hành")
    if card.trained_on_farm_data and target.upper() != (card.country or "").upper():
        raise BundleError("mô hình học từ dữ liệu trang trại không được gửi ra khỏi nước đó")
    payload = {
        "card": card.model_dump(mode="json", exclude={"history"}),
        "target": target.upper(),
        "issuer": issuer,
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    signed = canonical(payload)
    sig = _load_priv(private_pem).sign(signed)
    # "signed" = đúng các byte đã ký, để bên khác ngôn ngữ (Nexagri viết bằng Go) kiểm chữ ký
    # mà không phải tự dựng lại JSON chuẩn hoá giống hệt Python.
    return {"payload": payload, "signed": base64.b64encode(signed).decode(), "signature": base64.b64encode(sig).decode()}


def verify_bundle(bundle: dict, public_pem: bytes, country: str, artifact: Path | None = None) -> ModelCard:
    try:
        sig = base64.b64decode(bundle["signature"])
        if "signed" in bundle:
            signed = base64.b64decode(bundle["signed"])
            payload = json.loads(signed)
            if "payload" in bundle and bundle["payload"] != payload:
                raise ValueError("payload khác phần đã ký")
        else:
            payload = bundle["payload"]
            signed = canonical(payload)
        _load_pub(public_pem).verify(sig, signed)
    except (KeyError, ValueError, TypeError, InvalidSignature) as e:
        raise BundleError("chữ ký không hợp lệ: gói đã bị sửa hoặc không do Core phát hành") from e
    if payload.get("target", "").upper() != country.upper():
        raise BundleError(f"gói phát hành cho {payload.get('target')}, không phải {country}")
    card = ModelCard.model_validate(payload["card"])
    if gate_problems(card):
        raise BundleError("thẻ mô hình trong gói không qua cửa duyệt")
    if artifact is not None and file_sha256(artifact) != card.artifact_sha256:
        raise BundleError("tệp mô hình không khớp mã băm trong gói")
    return card
