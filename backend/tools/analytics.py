from langchain_core.tools import tool
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.loader import DATA


@tool
def customer_analytics(query_type: str) -> str:
    """
    Answers business intelligence questions using structured order data.

    Use this tool when the user asks about:
    - Revenue, sales, or payment data
    - Top customers by spend
    - Popular product categories
    - Geographic distribution of customers
    - Average order values
    - Payment methods

    Args:
        query_type: The type of analysis to run. Must be one of:
            - "top_customers" — top 10 customers by total spend
            - "revenue_by_category" — revenue breakdown by product category
            - "customer_by_city" — top cities by customer count
            - "payment_methods" — breakdown of payment types used
            - "avg_order_value" — average order value and payment stats
            - "order_status" — breakdown of order statuses

    Returns:
        A formatted analysis result as readable text.
    """
    orders = DATA["orders"]
    customers = DATA["customers"]
    order_items = DATA["order_items"]
    payments = DATA["payments"]
    products = DATA["products"]

    if query_type == "top_customers":
        # Join payments with orders to get customer spend
        customer_spend = (
            payments.merge(orders[["order_id", "customer_id"]], on="order_id")
            .groupby("customer_id")["payment_value"]
            .sum()
            .reset_index()
            .merge(customers[["customer_id", "customer_city"]], on="customer_id")
            .nlargest(10, "payment_value")
        )

        lines = ["Top 10 customers by total spend:\n"]
        for i, row in enumerate(customer_spend.itertuples(), 1):
            lines.append(
                f"{i}. Customer {row.customer_id[:8]}... "
                f"| Total Spend: R${row.payment_value:,.2f} "
                f"| City: {row.customer_city}"
            )
        return "\n".join(lines)

    elif query_type == "revenue_by_category":
        # Join order items with products for category revenue
        category_revenue = (
            order_items.merge(
                products[["product_id",
                           "product_category_name_english"]],
                on="product_id",
                how="left"
            )
            .groupby("product_category_name_english")["price"]
            .sum()
            .reset_index()
            .nlargest(10, "price")
        )
        category_revenue.columns = ["category", "revenue"]

        lines = ["Top 10 product categories by revenue:\n"]
        for i, row in enumerate(category_revenue.itertuples(), 1):
            lines.append(
                f"{i}. {row.category} "
                f"| Revenue: R${row.revenue:,.2f}"
            )
        return "\n".join(lines)

    elif query_type == "customer_by_city":
        city_counts = (
            customers.groupby("customer_city")
            .size()
            .reset_index(name="customer_count")
            .nlargest(10, "customer_count")
        )

        lines = ["Top 10 cities by customer count:\n"]
        for i, row in enumerate(city_counts.itertuples(), 1):
            lines.append(
                f"{i}. {row.customer_city.title()} "
                f"| Customers: {row.customer_count:,}"
            )
        return "\n".join(lines)

    elif query_type == "payment_methods":
        payment_breakdown = (
            payments.groupby("payment_type")
            .agg(
                count=("payment_value", "count"),
                total=("payment_value", "sum")
            )
            .reset_index()
            .sort_values("total", ascending=False)
        )

        lines = ["Payment method breakdown:\n"]
        for row in payment_breakdown.itertuples():
            lines.append(
                f"- {row.payment_type.replace('_', ' ').title()} "
                f"| Transactions: {row.count:,} "
                f"| Total Value: R${row.total:,.2f}"
            )
        return "\n".join(lines)

    elif query_type == "avg_order_value":
        avg_value = payments["payment_value"].mean()
        median_value = payments["payment_value"].median()
        total_revenue = payments["payment_value"].sum()
        total_orders = payments["order_id"].nunique()

        return (
            f"Order value statistics:\n"
            f"- Total Revenue: R${total_revenue:,.2f}\n"
            f"- Total Orders: {total_orders:,}\n"
            f"- Average Order Value: R${avg_value:,.2f}\n"
            f"- Median Order Value: R${median_value:,.2f}"
        )

    elif query_type == "order_status":
        status_counts = (
            orders.groupby("order_status")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        total = len(orders)

        lines = ["Order status breakdown:\n"]
        for row in status_counts.itertuples():
            pct = row.count / total * 100
            lines.append(
                f"- {row.order_status.title()} "
                f"| Count: {row.count:,} "
                f"| {pct:.1f}% of orders"
            )
        return "\n".join(lines)

    else:
        return (
            f"Unknown query_type: '{query_type}'. "
            f"Valid options: top_customers, revenue_by_category, "
            f"customer_by_city, payment_methods, avg_order_value, order_status"
        )