"""Suite-wide environment staging.

The gemm-v2 sidecar requirement defaults ON in production
(``VERATHOS_MESH_REQUIRE_GEMM_V2``); many fixtures in this suite write tiny
synthetic traces whose shapes the strict requirement rejects, exactly like
the capture-less fixtures that stage ``VERATHOS_MESH_REQUIRE_BOUNDARY_CHAIN``
off per-test. Stage the requirement off for the suite; required-mode
behavior is proven by the dedicated sidecar tests and the live e2e runs
(both require flags on and real hardware).
"""

import os

os.environ.setdefault("VERATHOS_MESH_REQUIRE_GEMM_V2", "0")
