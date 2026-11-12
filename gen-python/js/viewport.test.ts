import { columnsForContentWidth } from "./viewport";

function assertEqual(actual: unknown, expected: unknown, message: string): void {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

assertEqual(columnsForContentWidth(640, 8), 80, "wide content should use all fitting columns");
assertEqual(columnsForContentWidth(479, 8), 59, "narrowing the parent should reduce columns");
assertEqual(columnsForContentWidth(905, 8), 113, "widening the parent should increase columns");
assertEqual(columnsForContentWidth(7, 8), 1, "a positive width should retain one terminal column");
assertEqual(columnsForContentWidth(0, 8), null, "a hidden parent should defer resizing");
assertEqual(columnsForContentWidth(Number.NaN, 8), null, "invalid measurements should defer resizing");
