ALTER TABLE annotations
ADD COLUMN extra TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(extra));
