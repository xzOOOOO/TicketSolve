"""Agent 协作协议常量。"""

VALID_AGENTS = {"db_agent", "net_agent", "app_agent"}

PROTOCOL_MESSAGE_TYPES = {
    "hypothesis",
    "evidence_request",
    "evidence_response",
    "challenge",
    "support",
    "diagnosis",
}
