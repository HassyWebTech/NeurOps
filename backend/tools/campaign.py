from langchain_core.tools import tool
from groq import Groq
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, GROQ_MODEL
from data.loader import DATA
from memory.store import load_context


@tool
def generate_campaign(
    campaign_type: str = "reengagement",
    target_segment: str = "churning customers",
    tone: str = "friendly"
) -> str:
    """
    Generates a marketing campaign targeting specific customer segments.

    Use this tool when the user asks to:
    - Generate a campaign for inactive or churning customers
    - Write re-engagement messages or emails
    - Create WhatsApp messages for customers
    - Draft promotional offers for specific customer segments
    - Generate marketing copy based on customer data

    Args:
        campaign_type: Type of campaign to generate. Options:
            - "reengagement" — win back inactive customers
            - "loyalty" — reward and retain active customers
            - "promotional" — announce a sale or offer
        target_segment: Who the campaign targets.
            e.g. "churning customers", "top spenders", "new customers"
        tone: Communication tone.
            - "friendly" — warm and conversational
            - "urgent" — creates urgency, time-sensitive
            - "professional" — formal business tone

    Returns:
        A complete campaign package with WhatsApp message,
        email subject line, and strategic notes.
    """
    # Load business context — so the campaign reflects
    # what NeurOps knows about this specific business
    business_context = load_context()

    # Pull key business metrics to inform the campaign
    payments = DATA["payments"]
    orders = DATA["orders"]

    total_revenue = payments["payment_value"].sum()
    avg_order_value = payments["payment_value"].mean()
    total_customers = DATA["customers"]["customer_id"].nunique()
    delivered_rate = (
        orders["order_status"].eq("delivered").sum() / len(orders) * 100
    )

    # Format business context for the prompt
    context_lines = []
    for key, value in business_context.items():
        context_lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    context_text = "\n".join(context_lines) if context_lines else "Not specified"

    # Build the campaign generation prompt
    prompt = f"""You are a marketing expert specialising in customer retention 
for African businesses. Generate a complete campaign package.

Business Context:
{context_text}

Business Metrics:
- Total Customers: {total_customers:,}
- Average Order Value: R${avg_order_value:,.2f}
- Delivery Success Rate: {delivered_rate:.1f}%

Campaign Brief:
- Type: {campaign_type}
- Target Segment: {target_segment}
- Tone: {tone}

Generate a complete campaign package with exactly these sections:

1. WHATSAPP MESSAGE (max 160 characters, conversational, include emoji)
2. EMAIL SUBJECT LINE (max 60 characters, compelling)
3. EMAIL BODY (3-4 sentences, personalised, include a clear call to action)
4. DISCOUNT OFFER (specific percentage or value, with expiry timeframe)
5. STRATEGIC NOTES (2-3 bullet points on why this campaign will work)

Make it specific to the business context above. 
Do not use placeholder text like [Business Name] — 
use the actual business details provided."""

    # Call Groq directly for campaign generation
    # We use a higher temperature here (0.7) because
    # marketing copy benefits from creativity
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024
    )

    campaign = response.choices[0].message.content

    return f"Campaign Package ({campaign_type.title()} · {target_segment}):\n\n{campaign}"