FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
COPY --chown=user . /app
EXPOSE 7860
CMD ["python", "ui_nicegui.py"]