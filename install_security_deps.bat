@echo off
echo Installing RainLoom security packages...
pip install "email-validator>=2.1.0" "bcrypt>=4.1.0" "python-multipart>=0.0.9"
echo.
echo Done! You can now run: streamlit run monsoon_textile_app/app.py
pause
