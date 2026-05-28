SR_ERR_OPT_001 = "SR-ERR-OPT-001"
SR_ERR_OPT_002 = "SR-ERR-OPT-002"
SR_ERR_OPT_003 = "SR-ERR-OPT-003"
SR_ERR_OPT_004 = "SR-ERR-OPT-004"

SR_WARN_OPT_001 = "SR-WARN-OPT-001"
SR_WARN_OPT_002 = "SR-WARN-OPT-002"

SR_INV_STRUCT_001 = "SR-INV-STRUCT-001"
SR_INV_ARITY_001 = "SR-INV-ARITY-001"
SR_INV_OP_001 = "SR-INV-OP-001"
SR_INV_CONSTR_001 = "SR-INV-CONSTR-001"
SR_INV_CONSTR_002 = "SR-INV-CONSTR-002"
SR_INV_NEST_001 = "SR-INV-NEST-001"
SR_INV_EVAL_001 = "SR-INV-EVAL-001"
SR_INV_NONFINITE_001 = "SR-INV-NONFINITE-001"
SR_INV_OBJ_001 = "SR-INV-OBJ-001"

SR_ERR_TRACE_001 = "SR-ERR-TRACE-001"
SR_ERR_TRACE_002 = "SR-ERR-TRACE-002"
SR_ERR_TRACE_003 = "SR-ERR-TRACE-003"
SR_ERR_TRACE_004 = "SR-ERR-TRACE-004"


class BackendOptionError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class InvalidCandidateError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
