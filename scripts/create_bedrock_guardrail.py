"""Create the project's Amazon Bedrock guardrail once (a paid-account change you run yourself).

    BEDROCK_CALLS_ENABLED=true APP_PROFILE=bedrock python scripts/create_bedrock_guardrail.py

Prints the guardrail ID and version to put in .env.bedrock (GUARDRAIL_ID, GUARDRAIL_VERSION).
Needs an IAM identity allowed to call bedrock:CreateGuardrail and bedrock:CreateGuardrailVersion;
the running app only needs bedrock:ApplyGuardrail.
"""

from __future__ import annotations

from support_agents.aws import create_aws_client

BLOCK_INPUT = (
    "I can't help with that request. Please ask a question about school fees or academics."
)
BLOCK_OUTPUT = (
    "I can't share that answer. Please rephrase your question or contact the school office."
)


def main() -> None:
    client = create_aws_client("bedrock")
    created = client.create_guardrail(
        name="customer-support-guardrail",
        description="Prompt-attack, harmful-content, PII and off-topic checks for school support.",
        blockedInputMessaging=BLOCK_INPUT,
        blockedOutputsMessaging=BLOCK_OUTPUT,
        contentPolicyConfig={
            "filtersConfig": [
                {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
                {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            ]
        },
        topicPolicyConfig={
            "topicsConfig": [
                {
                    "name": "legal-and-medical-advice",
                    "definition": "Requests for legal, medical or financial investment advice.",
                    "examples": ["Should I sue the school?", "What medicine should my child take?"],
                    "type": "DENY",
                }
            ]
        },
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "ANONYMIZE"},
                {"type": "PASSWORD", "action": "ANONYMIZE"},
                {"type": "AWS_SECRET_KEY", "action": "BLOCK"},
            ]
        },
    )
    version = client.create_guardrail_version(guardrailIdentifier=created["guardrailId"])
    print(f"GUARDRAIL_ID={created['guardrailId']}")
    print(f"GUARDRAIL_VERSION={version['version']}")


if __name__ == "__main__":
    main()
