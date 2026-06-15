"""Model adapters — auto-register on import."""

# ── Action models (ModelAdapter) ──
from benchlink.models.rdp_adapter import RDPAdapter
from benchlink.models.fastwam_adapter import FastWAMAdapter
from benchlink.models.dp_adapter import DPAdapter
from benchlink.models.openpi_adapter import OpenPiAdapter
from benchlink.models.motus_adapter import MotusAdapter
from benchlink.models.dreamzero_adapter import DreamZeroAdapter
from benchlink.models.rdt_adapter import RDTAdapter
from benchlink.models.vla_touch_adapter import VLA_TouchAdapter

# ── Tactile encoders (TactileAdapter) ──
from benchlink.models.anytouch_adapter import AnyTouchAdapter
from benchlink.models.sparsh_adapter import SparshAdapter
from benchlink.models.t3_adapter import T3Adapter
from benchlink.models.unitac_ecf_adapter import UniTac_ECFAdapter

from benchlink.registry import register_model, register_tactile

# === Register action models ===
register_model("rdp", RDPAdapter)
register_model("fastwam", FastWAMAdapter)
register_model("dp", DPAdapter)
register_model("openpi", OpenPiAdapter)
register_model("motus", MotusAdapter)
register_model("dreamzero", DreamZeroAdapter)
register_model("vla_touch", VLA_TouchAdapter)
register_model("rdt", RDTAdapter)

# === Register tactile encoders ===
# Note: vla_touch is dual-registered (both ModelAdapter and TactileAdapter)
# because it implements both interfaces
register_tactile("anytouch", AnyTouchAdapter)
register_tactile("sparsh", SparshAdapter)
register_tactile("t3", T3Adapter)
register_tactile("vla_touch", VLA_TouchAdapter)
register_tactile("unitac_ecf", UniTac_ECFAdapter)
