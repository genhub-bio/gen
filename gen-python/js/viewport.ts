/** Return the number of terminal columns that fit in a content-box width. */
export function columnsForContentWidth(contentWidth: number, cellWidth: number): number | null {
  if (!Number.isFinite(contentWidth) || !Number.isFinite(cellWidth) || contentWidth <= 0 || cellWidth <= 0) {
    return null;
  }

  return Math.max(1, Math.floor(contentWidth / cellWidth));
}
