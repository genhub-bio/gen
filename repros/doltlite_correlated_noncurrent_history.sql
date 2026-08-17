.bail on
.open --new /tmp/doltlite-correlated-noncurrent-history.db

CREATE TABLE assets (
    id INTEGER PRIMARY KEY
);
SELECT dolt_commit('-A', '-m', 'create assets');

SELECT dolt_checkout('-b', 'feature');
INSERT INTO assets VALUES (1);
SELECT dolt_commit('-A', '-m', 'add feature row');
SELECT dolt_checkout('main');

-- Returns 1 when the history walk starts at the requested range.
WITH commit_range(start_ref) AS (
    SELECT dolt_hashof('feature')
)
SELECT assets.id
FROM commit_range
JOIN dolt_history_assets(commit_range.start_ref) AS assets;
