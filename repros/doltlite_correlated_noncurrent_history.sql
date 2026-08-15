.bail on
.echo on
.headers on
.mode box

.open --new /tmp/doltlite-correlated-noncurrent-history.db

CREATE TABLE assets (
    id INTEGER PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT INTO assets VALUES (1, 'main');
SELECT dolt_commit('-A', '-m', 'add main asset');

SELECT dolt_checkout('-b', 'feature');
INSERT INTO assets VALUES (2, 'feature');
SELECT dolt_commit('-A', '-m', 'add feature asset');
SELECT dolt_checkout('main');

-- Control: an uncorrelated commit_hash predicate reads the non-current feature commit.
SELECT count(*) AS direct_feature_rows
FROM dolt_history_assets
WHERE commit_hash = dolt_hashof('feature')
  AND id = 2;

-- Regression: affected DoltLite builds do not retarget the history walk when the equivalent
-- commit hash comes from a correlated join. They start at the session HEAD and return 0 here.
WITH RECURSIVE selected_commit(commit_hash) AS (
    SELECT dolt_hashof('feature')
    UNION ALL
    SELECT NULL FROM selected_commit WHERE FALSE
)
SELECT count(*) AS correlated_feature_rows
FROM selected_commit
JOIN dolt_history_assets AS historical_assets
  ON historical_assets.commit_hash = selected_commit.commit_hash
WHERE historical_assets.id = 2;

-- Keep the reproducer failable: this INSERT should succeed once the correlated constraint
-- starts the history walk at the selected commit. Affected builds fail CHECK(row_count = 1).
CREATE TEMP TABLE assertion (
    row_count INTEGER NOT NULL CHECK (row_count = 1)
);
INSERT INTO assertion
WITH RECURSIVE selected_commit(commit_hash) AS (
    SELECT dolt_hashof('feature')
    UNION ALL
    SELECT NULL FROM selected_commit WHERE FALSE
)
SELECT count(*)
FROM selected_commit
JOIN dolt_history_assets AS historical_assets
  ON historical_assets.commit_hash = selected_commit.commit_hash
WHERE historical_assets.id = 2;
