# Email Notifications Setup Guide

Your AI Stock Advisor can now send email notifications when training and backtesting complete!

## 🚀 Quick Setup

### 1. Enable Email Notifications
```bash
export EMAIL_ENABLED=true
```

### 2. Configure Email Settings
```bash
# Gmail example
export EMAIL_SMTP_SERVER="smtp.gmail.com"
export EMAIL_SMTP_PORT="587"
export EMAIL_USERNAME="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"
export EMAIL_TO="your-email@gmail.com"  # Where to send notifications
```

### 3. Gmail App Password (Important!)
For Gmail, you need an "App Password" (not your regular password):

1. Go to: https://myaccount.google.com/apppasswords
2. Enable 2-factor authentication if not already enabled
3. Click "Select app" → "Other (Custom name)"
4. Enter "AI Stock Advisor"
5. Click "Generate"
6. Copy the 16-character password (use this for EMAIL_PASSWORD)

### 4. Test the Setup
Run your training/backtesting - you'll receive emails when:
- ✅ Training completes (models trained, time taken, any failures)
- ✅ Backtesting completes (strategy results, time taken)
- ✅ Entire process finishes (summary with total runtime)
- ❌ Critical errors occur (automatic error notifications)

## 📧 Email Examples

### Training Completion Email
```
✅ SUCCESS: Training Complete

🤖 AI Stock Advisor - Training Complete

📅 Timestamp: 2026-01-09 15:30:45
✅ Status: SUCCESS

📊 Training Summary:
• Total Models: 835
• Successfully Trained: 832
• Failed: 3
• Success Rate: 99.6%
• Training Time: 45.2 minutes

❌ Failed Models:
• GEV: insufficient_data
• ABC: training_failed
• XYZ: memory_error

---
📧 This is an automated notification from AI Stock Advisor
🔧 To disable these notifications, set EMAIL_ENABLED=False
```

### Backtesting Completion Email
```
✅ SUCCESS: Backtesting Complete

🤖 AI Stock Advisor - Backtesting Complete

📅 Timestamp: 2026-01-09 16:15:30
✅ Status: SUCCESS

📊 Backtesting Summary:
• Backtest Time: 12.8 minutes
• Strategies Tested: 8

📈 Strategy Results:
• AI Strategy: 23.45% return
• Buy & Hold: 18.23% return
• Dynamic BH 1Y: 21.67% return

---
📧 This is an automated notification from AI Stock Advisor
🔧 To disable these notifications, set EMAIL_ENABLED=False
```

## 🔧 Other Email Providers

### Outlook/Hotmail
```bash
export EMAIL_SMTP_SERVER="smtp-mail.outlook.com"
export EMAIL_SMTP_PORT="587"
```

### Yahoo Mail
```bash
export EMAIL_SMTP_SERVER="smtp.mail.yahoo.com"
export EMAIL_SMTP_PORT="587"
```

### Custom SMTP
```bash
export EMAIL_SMTP_SERVER="your-smtp-server.com"
export EMAIL_SMTP_PORT="587"  # or 465 for SSL
```

## 🛡️ Security Tips

1. **Never hardcode passwords** - always use environment variables
2. **Use App Passwords** for Gmail, not your main password
3. **Set EMAIL_TO** to yourself or a trusted recipient
4. **Disable notifications** when not needed: `export EMAIL_ENABLED=false`

## 📱 Mobile Notifications

To get mobile notifications:
1. Set up email as above
2. Use your phone's email app with notifications enabled
3. You'll get instant alerts when training completes!

## 🚨 Troubleshooting

**"Authentication failed"**
- Check EMAIL_USERNAME and EMAIL_PASSWORD
- For Gmail, ensure you're using an App Password

**"Connection refused"**
- Check EMAIL_SMTP_SERVER and EMAIL_SMTP_PORT
- Ensure firewall allows SMTP traffic

**No emails received**
- Check spam/junk folder
- Verify EMAIL_TO address is correct
- Check EMAIL_ENABLED=true

## 🎯 Pro Tips

1. **Set different EMAIL_TO** to send notifications to a team
2. **Use a dedicated email** for automated notifications
3. **Monitor training progress** remotely without checking the terminal
4. **Get instant alerts** when your strategies are ready for analysis

That's it! You'll now get email notifications whenever your AI Stock Advisor completes training or backtesting! 🚀
