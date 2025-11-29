-- Drop all tables and recreate with pgvector support
-- Run this script: psql postgresql://postgres:postgres@localhost:5432/the_gathering_test -f scripts/reset_db_with_pgvector.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop all tables (CASCADE removes dependencies)
DROP TABLE IF EXISTS message_translations CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversation_participants CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS ai_memories CASCADE;
DROP TABLE IF EXISTS ai_cooldowns CASCADE;
DROP TABLE IF EXISTS ai_entities CASCADE;
DROP TABLE IF EXISTS room_users CASCADE;
DROP TABLE IF EXISTS rooms CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Tables will be recreated by SQLAlchemy on next app start
-- with updated schema (user_id, vector embedding, new indexes)
