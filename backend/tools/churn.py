from langchain_core.tools import tool
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.loader import DATA


@tool
def churn_analysis(top_n: int = 10) -> str:
    """
    Identifies customers most at risk of churning based on
    purchase recency, frequency, and order completion rate.

    Use this tool when the user asks about:
    - Which customers are likely to stop buying
    - Customer retention or churn risk
    - At-risk customers
    - Customers who haven't purchased recently

    Args:
        top_n: Number of at-risk customers to return (default 10)

    Returns:
        A formatted list of at-risk customers with their churn scores
        and key metrics.
    """
    orders = DATA["orders"].copy()
    customers = DATA["customers"].copy()

    # Step 1: Only work with orders that have a purchase timestamp
    orders = orders.dropna(subset=["order_purchase_timestamp"])

    # Step 2: Calculate recency — days since last purchase
    # We use the most recent order date in the dataset as "today"
    # so the analysis is consistent regardless of when it runs
    reference_date = orders["order_purchase_timestamp"].max()

    recency = (
        orders.groupby("customer_id")["order_purchase_timestamp"]
        .max()
        .reset_index()
    )
    recency.columns = ["customer_id", "last_purchase"]
    recency["days_since_purchase"] = (
        reference_date - recency["last_purchase"]
    ).dt.days

    # Step 3: Calculate frequency — total orders per customer
    frequency = (
        orders.groupby("customer_id")
        .size()
        .reset_index(name="total_orders")
    )

    # Step 4: Calculate completion rate
    # Delivered orders / total orders — low rate = bad experience = churn risk
    orders["is_delivered"] = (
        orders["order_status"] == "delivered"
    ).astype(int)

    completion = (
        orders.groupby("customer_id")["is_delivered"]
        .mean()
        .reset_index(name="completion_rate")
    )

    # Step 5: Merge all signals into one DataFrame
    df = recency.merge(frequency, on="customer_id")
    df = df.merge(completion, on="customer_id")
    df = df.merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left"
    )

    # Step 6: Calculate churn score (0-100)
    # Higher score = higher churn risk
    # Each component contributes equally (33.3% weight each)

    # Normalize recency: more days = higher risk
    max_days = df["days_since_purchase"].max()
    df["recency_score"] = df["days_since_purchase"] / max_days * 100

    # Normalize frequency: fewer orders = higher risk
    max_orders = df["total_orders"].max()
    df["frequency_score"] = (1 - df["total_orders"] / max_orders) * 100

    # Completion rate: lower completion = higher risk
    df["completion_score"] = (1 - df["completion_rate"]) * 100

    # Final churn score — weighted average
    df["churn_score"] = (
        df["recency_score"] * 0.5 +      # Recency weighted highest
        df["frequency_score"] * 0.3 +    # Frequency second
        df["completion_score"] * 0.2     # Completion rate third
    ).round(1)

    # Step 7: Return top N at-risk customers
    top_churners = (
        df.nlargest(top_n, "churn_score")
        [["customer_id", "churn_score", "days_since_purchase",
          "total_orders", "completion_rate", "customer_city"]]
    )

    # Step 8: Format as readable text for the LLM
    lines = [f"Top {top_n} customers at risk of churning:\n"]

    for i, row in top_churners.iterrows():
        lines.append(
            f"{len(lines)}. Customer {row['customer_id'][:8]}... "
            f"| Churn Risk: {row['churn_score']}% "
            f"| Last Purchase: {int(row['days_since_purchase'])} days ago "
            f"| Orders: {int(row['total_orders'])} "
            f"| Delivery Rate: {row['completion_rate']:.0%} "
            f"| City: {row['customer_city']}"
        )

    # Add summary statistics
    avg_churn = df["churn_score"].mean()
    high_risk = (df["churn_score"] > 75).sum()
    lines.append(f"\nSummary:")
    lines.append(f"- Average churn risk across all customers: {avg_churn:.1f}%")
    lines.append(f"- Customers with >75% churn risk: {high_risk:,}")

    return "\n".join(lines)