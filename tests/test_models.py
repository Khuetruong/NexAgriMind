import json
from datetime import datetime, timezone

import pytest

from agrimind.models.registry import ModelCard, ModelRegistry, RegistryError, gate_problems
from agrimind.models.signing import BundleError, new_keypair, sign_bundle, verify_bundle


def _card(**kw):
    base = dict(
        id="d02-do-forecast", version="1.0.0", domain="D02", task="dự báo oxy hòa tan 6 giờ tới",
        license="Apache-2.0", split_by="farm", risk_level=2, artifact_sha256="a" * 64,
        metrics=[{"name": "mae_mg_l", "value": 0.6, "baseline": 0.9, "higher_is_better": False, "min_gain": 0.1}],
    )
    base.update(kw)
    return ModelCard(**base)


def test_gate_rules():
    assert "cần 1 chuyên gia" in " ".join(gate_problems(_card()))
    probs = " ".join(gate_problems(_card(license="AGPL-3.0", split_by="random", risk_level=3, auto_action=True,
                                         metrics=[{"name": "acc", "value": 0.8, "baseline": 0.8}])))
    for want in ["giấy phép", "chia ngẫu nhiên", "chưa tốt hơn mốc", "không được tự hành động", "cần 2 chuyên gia"]:
        assert want in probs, want
    assert "phải ghi nước" in " ".join(gate_problems(_card(trained_on_farm_data=True)))


def test_registry_flow_and_farm_data_stays(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register(_card(trained_on_farm_data=True, country="VN"), by="kỹ sư A")
    with pytest.raises(RegistryError):
        reg.register(_card(), by="x")  # trùng phiên bản
    with pytest.raises(RegistryError):
        reg.mark_published("d02-do-forecast", "1.0.0", "core", "VN")  # chưa duyệt
    card = reg.approve("d02-do-forecast", "1.0.0", "TS. Thủy sản", "D02")
    assert card.status.value == "approved"
    with pytest.raises(RegistryError):
        reg.mark_published("d02-do-forecast", "1.0.0", "core", "TH")
    assert reg.mark_published("d02-do-forecast", "1.0.0", "core", "VN").status.value == "published"
    assert [e.action for e in reg.get("d02-do-forecast", "1.0.0").history] == [
        "register", "approve", "gate_passed", "publish"]


def test_signed_bundle(tmp_path):
    priv, pub = new_keypair()
    other_priv, other_pub = new_keypair()
    art = tmp_path / "model.bin"
    art.write_bytes(b"weights")
    import hashlib
    card = _card(artifact_sha256=hashlib.sha256(b"weights").hexdigest(), status="approved",
                 approvals=[{"expert": "TS. B", "domain": "D02", "at": datetime.now(timezone.utc)}])
    with pytest.raises(BundleError):
        sign_bundle(_card(), "VN", priv)  # chưa duyệt
    bundle = sign_bundle(card, "vn", priv)
    assert verify_bundle(bundle, pub, "VN", art).key == "d02-do-forecast@1.0.0"
    with pytest.raises(BundleError):
        verify_bundle(bundle, other_pub, "VN")  # khoá khác
    with pytest.raises(BundleError):
        verify_bundle(bundle, pub, "TH")  # sai nước
    tampered = json.loads(json.dumps(bundle))
    tampered["payload"]["card"]["risk_level"] = 1
    with pytest.raises(BundleError):
        verify_bundle(tampered, pub, "VN")
    art.write_bytes(b"weights-doi")
    with pytest.raises(BundleError):
        verify_bundle(bundle, pub, "VN", art)
