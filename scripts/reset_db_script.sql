-- Drop all tables and recreate the database from scratch
-- Usage: psql postgresql://postgres:postgres@localhost:5432/thegathering -f scripts/reset_db_script.sql
-- Tables are recreated by SQLAlchemy (create_tables) on next app start

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Drop all tables (CASCADE removes dependencies)
DROP TABLE IF EXISTS message_translation CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversation_participants CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS ai_memories CASCADE;
DROP TABLE IF EXISTS ai_cooldowns CASCADE;
DROP TABLE IF EXISTS ai_entities CASCADE;
DROP TABLE IF EXISTS rooms CASCADE;
DROP TABLE IF EXISTS users CASCADE;
