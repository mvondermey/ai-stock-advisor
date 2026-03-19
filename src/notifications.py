"""
Email Notifications Module
Sends email notifications when training/backtesting completes.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from typing import Optional
import traceback
import requests

# Email configuration (set as environment variables for security)
EMAIL_ENABLED = os.environ.get('EMAIL_ENABLED', 'False').lower() == 'true'
EMAIL_SMTP_SERVER = os.environ.get('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.environ.get('EMAIL_SMTP_PORT', '587'))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
EMAIL_TO = os.environ.get('EMAIL_TO', EMAIL_USERNAME)  # Default to self if not specified

# ntfy.sh push notification configuration - imported from config.py
from config import NTFY_ENABLED, NTFY_TOPIC, NTFY_SERVER

def send_completion_notification(
    subject: str,
    message_body: str,
    success: bool = True,
    error_details: Optional[str] = None
) -> bool:
    """
    Send email notification about training/backtesting completion.
    
    Args:
        subject: Email subject line
        message_body: Main message content
        success: Whether the operation was successful
        error_details: Error details if failed
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not EMAIL_ENABLED:
        print("📧 Email notifications disabled (EMAIL_ENABLED=False)")
        return False
    
    if not all([EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO]):
        print("⚠️ Email configuration incomplete. Set EMAIL_USERNAME, EMAIL_PASSWORD, and EMAIL_TO environment variables.")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = EMAIL_TO
        msg['Subject'] = f"{'✅ SUCCESS' if success else '❌ FAILED'}: {subject}"
        
        # Build email body
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        email_body = f"""
🤖 AI Stock Advisor - {subject}

📅 Timestamp: {timestamp}
{'✅ Status: SUCCESS' if success else '❌ Status: FAILED'}

{message_body}

---
📧 This is an automated notification from AI Stock Advisor
🔧 To disable these notifications, set EMAIL_ENABLED=False
        """.strip()
        
        if error_details:
            email_body += f"\n\n🔍 Error Details:\n{error_details}"
        
        msg.attach(MIMEText(email_body, 'plain'))
        
        # Send email
        print(f"📧 Sending email notification to {EMAIL_TO}...")
        
        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print("✅ Email notification sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email notification: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def send_training_notification(
    models_trained: int,
    total_models: int,
    training_time_minutes: float,
    failed_models: Optional[dict] = None
) -> bool:
    """Send notification about training completion."""
    
    success_rate = (models_trained / total_models) * 100 if total_models > 0 else 0
    
    message_body = f"""
📊 Training Summary:
• Total Models: {total_models}
• Successfully Trained: {models_trained}
• Failed: {total_models - models_trained}
• Success Rate: {success_rate:.1f}%
• Training Time: {training_time_minutes:.1f} minutes
    """.strip()
    
    if failed_models:
        message_body += f"\n\n❌ Failed Models:\n"
        for ticker, reason in list(failed_models.items())[:10]:  # Show first 10
            message_body += f"• {ticker}: {reason}\n"
        if len(failed_models) > 10:
            message_body += f"... and {len(failed_models) - 10} more\n"
    
    return send_completion_notification(
        subject="Training Complete",
        message_body=message_body,
        success=models_trained > 0
    )

def send_backtesting_notification(
    strategy_results: dict,
    backtest_time_minutes: float,
    error_details: Optional[str] = None
) -> bool:
    """Send notification about backtesting completion."""
    
    # Send push notification first
    if error_details:
        send_push_notification(
            title="❌ Backtest Failed",
            message=f"Backtest failed after {backtest_time_minutes:.1f} min - check logs",
            priority="high",
            tags="warning"
        )
    else:
        # Get top strategy
        top_strategy = None
        top_return = None
        if strategy_results:
            for s, r in strategy_results.items():
                if isinstance(r, dict) and 'return' in r:
                    ret = r['return']
                    if top_return is None or ret > top_return:
                        top_return = ret
                        top_strategy = s
        send_push_notification(
            title="✅ Backtest Complete",
            message=f"Completed in {backtest_time_minutes:.1f} min" + (f" | Top: {top_strategy} {top_return:.1f}%" if top_strategy else ""),
            priority="default",
            tags="white_check_mark"
        )
    
    message_body = f"""
📊 Backtesting Summary:
• Backtest Time: {backtest_time_minutes:.1f} minutes
• Strategies Tested: {len(strategy_results)}
    """.strip()
    
    if strategy_results:
        message_body += "\n\n📈 Strategy Results:\n"
        for strategy, result in strategy_results.items():
            if isinstance(result, dict) and 'return' in result:
                message_body += f"• {strategy}: {result['return']:.2f}% return\n"
    
    return send_completion_notification(
        subject="Backtesting Complete",
        message_body=message_body,
        success=error_details is None,
        error_details=error_details
    )

def send_error_notification(
    error_type: str = None,
    error_message: str = None,
    traceback_str: str = None,
    operation: str = None,
    error: Exception = None,
    context: Optional[str] = None
) -> bool:
    """Send notification about critical errors."""
    
    # Support both old and new calling conventions
    if error_type and error_message:
        # New style: called with error_type, error_message, traceback_str
        op = operation or "Backtesting"
        error_details = f"Error Type: {error_type}\n"
        error_details += f"Error Message: {error_message}\n"
        if traceback_str:
            error_details += f"Traceback:\n{traceback_str}"
    else:
        # Old style: called with operation, error, context
        op = operation or "Unknown"
        error_details = f"Operation: {op}\n"
        if context:
            error_details += f"Context: {context}\n"
        if error:
            error_details += f"Error: {str(error)}\n"
        error_details += f"Traceback:\n{traceback.format_exc()}"
    
    # Send push notification
    send_push_notification(
        title=f"❌ Error: {op}",
        message=error_message or str(error) or "Unknown error",
        priority="high",
        tags="warning"
    )
    
    return send_completion_notification(
        subject=f"Critical Error: {op}",
        message_body=f"❌ A critical error occurred during {op}",
        success=False,
        error_details=error_details
    )

# =============================================================================
# ntfy.sh Push Notifications (easiest - no account needed)
# =============================================================================

def send_push_notification(
    title: str,
    message: str,
    priority: str = "default",
    tags: str = None
) -> bool:
    """
    Send push notification via ntfy.sh.
    
    Setup:
    1. Install ntfy app on your phone (iOS/Android)
    2. Subscribe to your topic (e.g., "ai-stock-advisor-yourname")
    3. Set environment variables:
       export NTFY_ENABLED=true
       export NTFY_TOPIC=ai-stock-advisor-yourname
    
    Args:
        title: Notification title
        message: Notification body
        priority: "min", "low", "default", "high", "urgent"
        tags: Comma-separated emoji tags (e.g., "warning,skull")
    
    Returns:
        bool: True if sent successfully
    """
    print(f"   [DEBUG] Push notification: NTFY_ENABLED={NTFY_ENABLED}, NTFY_TOPIC={NTFY_TOPIC}")
    if not NTFY_ENABLED:
        print(f"   [WARN] Push notifications disabled (NTFY_ENABLED=false)")
        return False
    
    try:
        url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
        # Encode title to handle emojis (HTTP headers must be ASCII/latin-1)
        import urllib.parse
        encoded_title = urllib.parse.quote(title, safe='')
        headers = {
            "Title": encoded_title,
            "Priority": priority,
            "Encoding": "utf-8",
        }
        if tags:
            headers["Tags"] = tags
        
        response = requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
        
        if response.status_code == 200:
            print(f"📱 Push notification sent to topic '{NTFY_TOPIC}'")
            return True
        else:
            print(f"⚠️ Push notification failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️ Push notification error: {e}")
        return False

def send_push_success(backtest_time_minutes: float, top_strategy: str = None, top_return: float = None):
    """Send success push notification with summary."""
    message = f"Backtest completed in {backtest_time_minutes:.1f} min"
    if top_strategy and top_return is not None:
        message += f"\n🏆 Best: {top_strategy} ({top_return:+.1f}%)"
    
    send_push_notification(
        title="✅ Backtest Complete",
        message=message,
        priority="default",
        tags="chart_with_upwards_trend,white_check_mark"
    )

def send_push_error(error_type: str, error_message: str):
    """Send error push notification."""
    send_push_notification(
        title=f"ERROR: {error_type}",
        message=error_message[:200],  # Truncate long messages
        priority="high",
        tags="warning,skull"
    )

def send_push_summary(backtest_time_minutes: float, strategy_returns: list):
    """
    Send push notification with full strategy summary.
    
    Args:
        backtest_time_minutes: Total backtest runtime
        strategy_returns: List of (strategy_name, return_pct) tuples, sorted by return
    """
    # Build message with top strategies
    lines = [f"⏱️ {backtest_time_minutes:.1f} min", ""]
    
    # Strategy name mapping for shorter display
    name_map = {
        'momentum_volatility_hybrid_6m': 'Mom-Vol 6M',
        'momentum_volatility_hybrid_1y': 'Mom-Vol 1Y',
        'momentum_volatility_hybrid': 'Mom-Vol 3M',
        'momentum_volatility_hybrid_1y3m': 'Mom-Vol 1Y/3M',
        'static_bh_1y': 'Static BH 1Y',
        'static_bh_6m': 'Static BH 6M',
        'static_bh_3m': 'Static BH 3M',
        'static_bh_1m': 'Static BH 1M',
        'dynamic_bh_1y': 'Dynamic BH 1Y',
        'dynamic_bh_6m': 'Dynamic BH 6M',
        'dynamic_bh_3m': 'Dynamic BH 3M',
        'dynamic_bh_1m': 'Dynamic BH 1M',
        'risk_adj_mom': 'Risk-Adj Mom',
        'risk_adj_mom_6m': 'Risk-Adj 6M',
        'risk_adj_mom_3m': 'Risk-Adj 3M',
        'risk_adj_mom_1m': 'Risk-Adj 1M',
        'elite_hybrid': 'Elite Hybrid',
        'elite_risk': 'Elite Risk',
        'ai_elite': 'AI Elite',
        'ai_elite_monthly': 'AI Elite Mth',
        'trend_atr': 'Trend ATR',
        'dual_momentum': 'Dual Mom',
        'price_acceleration': 'Price Accel',
        'enhanced_volatility': 'Enh Vol',
        'inverse_etf_hedge': '🛡️ Inv ETF',
    }
    
    for i, (name, ret) in enumerate(strategy_returns[:10], 1):
        display_name = name_map.get(name, name.replace('_', ' ').title()[:12])
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        lines.append(f"{emoji} {display_name}: {ret:+.1f}%")
    
    message = "\n".join(lines)
    
    send_push_notification(
        title="✅ Backtest Complete",
        message=message,
        priority="default",
        tags="chart_with_upwards_trend"
    )
