FROM mcr.microsoft.com/playwright/python:v1.41.0-jammy

WORKDIR /app

RUN pip install --no-cache-dir \
    playwright==1.41.0 \
    beautifulsoup4>=4.12.0 \
    lxml>=5.1.0 \
    pydantic>=2.5.0 \
    redis[hiredis]>=5.0.0 \
    httpx>=0.26.0 \
    structlog>=24.1.0 \
    tenacity>=8.2.0

COPY playwright_scraper/ ./playwright_scraper/
COPY normalization/ ./normalization/

# Non-root scraping user
RUN groupadd -r scraper && useradd -r -g scraper scraper && \
    chown -R scraper:scraper /app
USER scraper

CMD ["python", "-m", "playwright_scraper.runner"]
