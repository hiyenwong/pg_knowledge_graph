# ============================================================
# Stage 1: Build pg_knowledge_graph extension
# ============================================================
FROM rust:slim-bookworm AS builder

# Install PostgreSQL 17 dev headers + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release ca-certificates \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
       | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] \
       http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
       > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        postgresql-server-dev-17 \
        clang \
        llvm \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-pgrx (version must match Cargo.toml exactly)
RUN cargo install cargo-pgrx --version "=0.17.0" --locked

# Initialize pgrx against the system PG17
RUN cargo pgrx init --pg17 /usr/bin/pg_config

WORKDIR /build

# Copy manifests first so dependency layer is cached
COPY Cargo.toml ./
# Cargo.lock is .gitignore'd (library convention); generate it if absent
RUN test -f Cargo.lock || cargo generate-lockfile

# Copy full source and build
COPY . .

RUN cargo pgrx package --no-default-features --features pg17

# ============================================================
# Stage 2: Runtime — PostgreSQL 17 with pgvector pre-installed
# ============================================================
FROM pgvector/pgvector:pg17

# Copy compiled extension files from builder
COPY --from=builder \
    /build/target/release/pg_knowledge_graph-pg17/usr/share/postgresql/17/extension/ \
    /usr/share/postgresql/17/extension/

COPY --from=builder \
    /build/target/release/pg_knowledge_graph-pg17/usr/lib/postgresql/17/lib/ \
    /usr/lib/postgresql/17/lib/

# Default DB for testing
ENV POSTGRES_DB=kg_test
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Auto-run init script on first start
COPY docker/init.sql /docker-entrypoint-initdb.d/01_init.sql
