FROM rust:1.78-slim-bookworm AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY airpulse-types/ airpulse-types/
COPY airpulse-store/ airpulse-store/
COPY airpulse-classify/ airpulse-classify/
COPY airpulse-ingest/ airpulse-ingest/
COPY airpulse-enrich/ airpulse-enrich/
COPY airpulse-baseline/ airpulse-baseline/
COPY airpulse-digest/ airpulse-digest/
COPY airpulse-jira/ airpulse-jira/
COPY airpulse-scrape/ airpulse-scrape/
COPY airpulse-ws/ airpulse-ws/
COPY airpulse-notify/ airpulse-notify/
COPY airpulse-api/ airpulse-api/
RUN cargo build --release --bin airpulse-api 2>/dev/null || cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y python3 python3-pip curl ca-certificates \
    && pip3 install --break-system-packages scrapling \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /build/target/release/airpulse* /app/
COPY scrapers/ /app/scrapers/
EXPOSE 8741
CMD ["/app/airpulse-api"]
