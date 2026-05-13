ALTER TABLE samples ADD COLUMN is_reference INTEGER NOT NULL DEFAULT 0 CHECK (is_reference IN (0, 1));
