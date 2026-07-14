"""Re-export every prompt template version.

Many of these versions are referenced only via `version2template` in
`prompts/library.py` (a name -> template dict), so static analysis can't
see the use site. Listing each name in __all__ makes the re-export
explicit so ruff F401 doesn't flag them.
"""

from .template_cc_v1 import CC_TEMPLATE_V1
from .template_cc_v2 import CC_TEMPLATE_V2
from .template_codegen_v1 import CODEGEN_TEMPLATE_V1
from .template_codegen_v2 import CODEGEN_TEMPLATE_V2
from .template_codegen_v3 import CODEGEN_TEMPLATE_V3
from .template_codegen_v4 import CODEGEN_TEMPLATE_V4
from .template_codegen_v5 import CODEGEN_TEMPLATE_V5
from .template_codegen_v6 import CODEGEN_TEMPLATE_V6
from .template_codegen_v7 import CODEGEN_TEMPLATE_V7
from .template_codegen_v8 import CODEGEN_TEMPLATE_V8
from .template_codegen_v9 import CODEGEN_TEMPLATE_V9
from .template_codegen_v10 import CODEGEN_TEMPLATE_V10
from .template_codegen_v11 import CODEGEN_TEMPLATE_V11
from .template_final_prediction_v1 import FINAL_PREDICTION_TEMPLATE_V1
from .template_final_prediction_v2 import FINAL_PREDICTION_TEMPLATE_V2
from .template_final_prediction_v3 import FINAL_PREDICTION_TEMPLATE_V3
from .template_final_prediction_v4 import FINAL_PREDICTION_TEMPLATE_V4
from .template_final_prediction_v5 import FINAL_PREDICTION_TEMPLATE_V5
from .template_icl_v1 import ICL_TEMPLATE_V1
from .template_icl_v2 import ICL_TEMPLATE_V2
from .template_react_feedback_v1 import REACT_FEEDBACK_TEMPLATE_V1
from .template_react_feedback_v2 import REACT_FEEDBACK_TEMPLATE_V2
from .template_react_v1 import REACT_TEMPLATE_V1
from .template_react_v2 import REACT_TEMPLATE_V2
from .template_react_v3 import REACT_TEMPLATE_V3
from .template_rulebook_v1 import RULEBOOK_TEMPLATE_V1
from .template_rulebook_v2 import RULEBOOK_TEMPLATE_V2
from .template_v1 import AGENT_TEMPLATE_V1
from .template_v2 import AGENT_TEMPLATE_V2
from .template_v3 import AGENT_TEMPLATE_V3
from .template_v4 import AGENT_TEMPLATE_V4
from .template_v5 import AGENT_TEMPLATE_V5
from .template_v6 import AGENT_TEMPLATE_V6

__all__ = [
    "CC_TEMPLATE_V1",
    "CC_TEMPLATE_V2",
    "CODEGEN_TEMPLATE_V1",
    "CODEGEN_TEMPLATE_V2",
    "CODEGEN_TEMPLATE_V3",
    "CODEGEN_TEMPLATE_V4",
    "CODEGEN_TEMPLATE_V5",
    "CODEGEN_TEMPLATE_V6",
    "CODEGEN_TEMPLATE_V7",
    "CODEGEN_TEMPLATE_V8",
    "CODEGEN_TEMPLATE_V9",
    "CODEGEN_TEMPLATE_V10",
    "CODEGEN_TEMPLATE_V11",
    "FINAL_PREDICTION_TEMPLATE_V1",
    "FINAL_PREDICTION_TEMPLATE_V2",
    "FINAL_PREDICTION_TEMPLATE_V3",
    "FINAL_PREDICTION_TEMPLATE_V4",
    "FINAL_PREDICTION_TEMPLATE_V5",
    "ICL_TEMPLATE_V1",
    "ICL_TEMPLATE_V2",
    "REACT_FEEDBACK_TEMPLATE_V1",
    "REACT_FEEDBACK_TEMPLATE_V2",
    "REACT_TEMPLATE_V1",
    "REACT_TEMPLATE_V2",
    "REACT_TEMPLATE_V3",
    "RULEBOOK_TEMPLATE_V1",
    "RULEBOOK_TEMPLATE_V2",
    "AGENT_TEMPLATE_V1",
    "AGENT_TEMPLATE_V2",
    "AGENT_TEMPLATE_V3",
    "AGENT_TEMPLATE_V4",
    "AGENT_TEMPLATE_V5",
    "AGENT_TEMPLATE_V6",
]
