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

# Email configuration (set as environment variables for security)
EMAIL_ENABLED = os.environ.get('EMAIL_ENABLED', 'False').lower() == 'true'
EMAIL_SMTP_SERVER = os.environ.get('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.environ.get('EMAIL_SMTP_PORT', '587'))
EMAIL_USERNAME = os.environ.get('EMAIL_USERNAME', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
EMAIL_TO = os.environ.get('EMAIL_TO', EMAIL_USERNAME)  # Default to self if not specified

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
    operation: str,
    error: Exception,
    context: Optional[str] = None
) -> bool:
    """Send notification about critical errors."""
    
    error_details = f"Operation: {operation}\n"
    if context:
        error_details += f"Context: {context}\n"
    error_details += f"Error: {str(error)}\n"
    error_details += f"Traceback:\n{traceback.format_exc()}"
    
    return send_completion_notification(
        subject=f"Critical Error: {operation}",
        message_body=f"❌ A critical error occurred during {operation}",
        success=False,
        error_details=error_details
    )
