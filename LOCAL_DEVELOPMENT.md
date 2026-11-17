# Local development environment

This setup is isolated from Railway and uses only loopback-bound services. The
Compose project has unique names and does not stop or remove unrelated Docker
resources.

## First-time setup

1. Install and start Docker Desktop.
2. Copy `backend/.env.local.example` to `backend/.env.local` and set a local
   session secret. Keep any existing Groq key configuration in `backend/.env`
   private; never commit it.
3. Copy `frontend/.env.local.example` to `frontend/.env.local`.
4. Start infrastructure:

   ```powershell
   docker compose -f docker-compose.local.yml up -d
   ```

5. Apply the existing migrations to the local development database only:

   ```powershell
   $env:DATABASE_URL = "postgresql+asyncpg://pontis_local:pontis_local_dev_only@127.0.0.1:55432/pontis_eve_local"
   Get-ChildItem backend/migrations/*.sql | Sort-Object Name | ForEach-Object {
     docker exec -i pontis-eve-local-postgres psql -U pontis_local -d pontis_eve_local -v ON_ERROR_STOP=1 < $_.FullName
   }
   ```

   The migration files are plain SQL and are applied in filename order. The
   test database is created automatically, but is intentionally not populated
   by the development migration command.

## Run the application

Use a separate PowerShell window for the backend. Load the local values and,
if required, add the existing local Groq keys without printing them:

```powershell
Get-Content backend/.env.local | ForEach-Object {
  if ($_ -match '^([^#=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend in another window:

```powershell
Set-Location frontend
yarn install --frozen-lockfile
yarn dev --host 127.0.0.1
```

## Safety checks

Before migrations or integration tests, verify the resolved URLs contain
`127.0.0.1` and the expected local database names. Do not use `backend/.env`
as a migration environment while it contains Railway values. The repository
`.gitignore` already ignores `.env`, `.env.local`, and other environment files.

Useful checks:

```powershell
docker compose -f docker-compose.local.yml ps
docker exec pontis-eve-local-postgres pg_isready -U pontis_local -d pontis_eve_local
docker exec pontis-eve-local-redis redis-cli ping
Invoke-WebRequest http://127.0.0.1:56333/readyz
```

The real Groq suite remains opt-in and consumes provider quota. Run it only
after confirming the process has local database and Redis URLs:

```powershell
$env:EVE_INTEGRATION_TESTS = "1"
$env:TEST_DATABASE_URL = "postgresql+asyncpg://pontis_local:pontis_local_dev_only@127.0.0.1:55432/pontis_eve_test"
$env:TEST_REDIS_URL = "redis://127.0.0.1:56379/1"
pytest -n 0 -m real_llm tests/integration/test_eve_real_llm_chat.py -s
```
