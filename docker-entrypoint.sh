#!/usr/bin/env sh
# Apply database migrations, then exec the container command (the API server).
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting: $*"
exec "$@"
