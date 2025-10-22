import smtplib

sender_email = "pratikksecondaryacc@gmail.com"      # 🔹 replace with your Gmail
app_password = "xtbzqqbisteqltzw"         # 🔹 replace with your 16-digit App Password

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, app_password)
        print("✅ Login successful — App Password works!")
except Exception as e:
    print("❌ Login failed:", e)
