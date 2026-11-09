"""Exercise GraphWidget.go_to()/show() with Position and SuperPosition targets."""

import random

import gen

from test_api import RepositoryTestCase, FIXTURES

ACCENT_CURSOR = "#89b4fa"
HIGHLIGHT = "#f9e2af"
NODE_BACKGROUND = "#cdd6f4"


class WidgetNavigationTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        [self.locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.widget = self.graph.plot()

    def test_go_to_accepts_a_position(self):
        self.widget.go_to(self.locus.start())
        self.widget.go_to(self.locus.start(), center=True)

    def test_show_hides_cursor_until_shown_again(self):
        self.widget.show(self.locus, center=True)
        self.assertFalse(
            any(cell.get("bg") == ACCENT_CURSOR for cell in self.widget.frame["cells"])
        )

        self.widget.show_cursor()
        self.assertTrue(
            any(cell.get("bg") == ACCENT_CURSOR for cell in self.widget.frame["cells"])
        )

    def test_show_accepts_a_position(self):
        self.widget = self.graph.plot(rows=16, cols=90, detail="full").hide_cursor()
        self.widget.go_to(self.locus.start(), center=True)
        before = {(cell["x"], cell["y"]): cell for cell in self.widget.frame["cells"]}

        self.widget.show(self.locus.start(), color="cyan", center=True)
        changed = [
            cell
            for cell in self.widget.frame["cells"]
            if cell.get("bg") != before.get((cell["x"], cell["y"]), {}).get("bg")
        ]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["text"], self.locus.sequence[0])
        self.assertEqual(changed[0]["bg"], "#00cccc")

    def test_go_to_accepts_a_superposition(self):
        arm_ends = gen.SuperPosition(self.locus.start(), self.locus.end())

        self.widget.go_to(arm_ends)
        self.widget.go_to(arm_ends, center=True)

    def test_show_accepts_a_superposition(self):
        self.widget = self.graph.plot(rows=16, cols=90, detail="full").hide_cursor()
        arm_ends = gen.SuperPosition(self.locus.start(), self.locus.end())
        self.widget.go_to(self.locus.start(), center=True)
        before = {(cell["x"], cell["y"]): cell for cell in self.widget.frame["cells"]}

        self.widget.show(arm_ends, color="cyan", center=True)
        changed = [
            cell
            for cell in self.widget.frame["cells"]
            if cell.get("bg") != before.get((cell["x"], cell["y"]), {}).get("bg")
        ]
        self.assertEqual(len(changed), 2)
        self.assertEqual(
            "".join(
                cell["text"] for cell in sorted(changed, key=lambda cell: cell["x"])
            ),
            self.locus.sequence[0] + self.locus.sequence[-1],
        )
        self.assertEqual({cell["bg"] for cell in changed}, {"#00cccc"})

    def test_show_locus_highlights_only_its_sequence_cells(self):
        self.widget = self.graph.plot(rows=16, cols=90, detail="full").hide_cursor()
        self.widget.go_to(self.locus, center=True)
        before = {(cell["x"], cell["y"]): cell for cell in self.widget.frame["cells"]}

        self.widget.show(self.locus, color="cyan", center=True)
        after = {(cell["x"], cell["y"]): cell for cell in self.widget.frame["cells"]}
        changed = [
            position
            for position, cell in after.items()
            if cell.get("bg") != before.get(position, {}).get("bg")
        ]
        self.assertEqual(len(changed), len(self.locus))
        self.assertEqual({position[1] for position in changed}, {changed[0][1]})
        self.assertEqual(
            "".join(after[position]["text"] for position in sorted(changed)),
            self.locus.sequence,
        )

        self.widget.clear_highlights()
        self.assertEqual(self.widget.frame["cells"], list(before.values()))

    def test_show_annotation_highlights_its_sequence_cells(self):
        self.widget = self.graph.plot(rows=16, cols=90, detail="full").hide_cursor()
        annotation = gen.Annotation(self.locus, "temporary")

        self.widget.show(annotation, color="yellow", center=True)
        highlighted_bases = [
            cell
            for cell in self.widget.frame["cells"]
            if cell["text"] in "ACGT" and cell.get("bg") == "#cccc00"
        ]
        self.assertEqual(
            "".join(
                cell["text"]
                for cell in sorted(highlighted_bases, key=lambda cell: cell["x"])
            ),
            self.locus.sequence,
        )
        self.assertEqual(
            {cell["y"] for cell in highlighted_bases},
            {highlighted_bases[0]["y"]},
        )

    def test_show_hidden_locus_highlights_only_truncated_ellipsis(self):
        graph = self.repository.import_sequence(
            "A" * 24 + "CGTACGTT" + "A" * 24,
            name="truncated_highlight",
        )
        [locus] = graph.search("CGTACGTT", sequence_kind="exact")
        widget = graph.plot(rows=16, cols=90, detail="normal").hide_cursor()
        widget.go_to(locus, center=True)
        before = {(cell["x"], cell["y"]): cell for cell in widget.frame["cells"]}

        widget.show(locus, color="cyan", center=True)
        changed = [
            cell
            for cell in widget.frame["cells"]
            if cell.get("bg") != before.get((cell["x"], cell["y"]), {}).get("bg")
        ]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["text"], "\u2026")

    def test_annotation_widens_truncated_node_without_losing_final_base(self):
        graph = self.repository.import_sequence("abcdefgh", name="foo")
        [locus] = graph.search("cde", sequence_kind="exact")
        graph.add_annotation(locus, "bar")

        widget = graph.plot().show_labels()
        cells = widget.frame["cells"]
        self.assertIn("bar", "".join(cell["text"] for cell in cells))
        self.assertIn("h", [cell["text"] for cell in cells])


class WidgetLabelTests(RepositoryTestCase):
    """Plots open with annotation names hidden; show_labels() and zooming reveal them."""

    NAME = "long_annotation_name"

    def setUp(self):
        super().setUp()
        random.seed(5)
        sequence = "".join(random.choice("ACGT") for _ in range(120))
        self.graph = self.repository.import_sequence(sequence, name="labelled")
        self.locus = self.graph.region("labelled:40-60")
        self.graph.add_annotation(self.locus, self.NAME)
        self.widget = self.graph.plot(rows=16, cols=90, detail="normal").hide_cursor()

    def text(self):
        return "".join(cell["text"] for cell in self.widget.frame["cells"])

    def test_plots_open_with_names_hidden_and_labels_toggle_them(self):
        self.assertNotIn(self.NAME, self.text())
        hidden = self.text()

        self.assertIs(self.widget.show_labels(), self.widget)
        self.assertIn(self.NAME, self.text())

        self.assertIs(self.widget.hide_labels(), self.widget)
        self.assertEqual(self.text(), hidden)

    def test_zoom_steps_from_hidden_names_to_shown_names_to_full_sequence(self):
        self.assertNotIn(self.NAME, self.text())
        self.widget.zoom_in()
        self.assertIn(self.NAME, self.text())
        self.widget.zoom_in()
        self.assertIn(self.NAME, self.text())
        self.assertIn(self.locus.sequence, self.text().replace(".", ""))

        self.widget.zoom_out().zoom_out()
        self.assertNotIn(self.NAME, self.text())

    def test_show_highlights_the_annotation_cells_while_names_are_hidden(self):
        before = {(cell["x"], cell["y"]): cell for cell in self.widget.frame["cells"]}

        self.widget.show(self.graph.annotations[0], color="yellow", center=True)
        self.assertNotIn(self.NAME, self.text())
        highlighted = [
            cell
            for cell in self.widget.frame["cells"]
            if cell.get("bg") == "#cccc00"
            and cell.get("bg") != before.get((cell["x"], cell["y"]), {}).get("bg")
        ]
        self.assertTrue(highlighted, "should highlight the annotation without its name")

        self.widget.show_labels()
        still_highlighted = [
            cell for cell in self.widget.frame["cells"] if cell.get("bg") == "#cccc00"
        ]
        self.assertTrue(still_highlighted)
        self.assertIn(self.NAME, self.text())


class WidgetCursorAndZoomTests(RepositoryTestCase):
    """The cursor marks navigation targets, and zooming keeps the view on them."""

    NEEDLE_START = 700

    def setUp(self):
        super().setUp()
        random.seed(3)
        self.sequence = "".join(random.choice("ACGT") for _ in range(1500))
        graph = self.graph = self.repository.import_sequence(self.sequence, name="long")
        self.needle = self.sequence[self.NEEDLE_START : self.NEEDLE_START + 12]
        [self.locus] = graph.search(self.needle, sequence_kind="exact")
        self.widget = graph.plot(rows=10, cols=90)

    def cells(self):
        return {(cell["x"], cell["y"]): cell for cell in self.widget.frame["cells"]}

    def cursor_cells(self):
        return [
            position
            for position, cell in self.cells().items()
            if cell.get("bg") == ACCENT_CURSOR
        ]

    def arrows(self):
        return [
            position
            for position, cell in self.cells().items()
            if cell["text"] == "\u2193"
        ]

    def row_text(self, y):
        cells = self.cells()
        return "".join(cells.get((x, y), {}).get("text") or " " for x in range(90))

    def test_go_to_draws_an_accent_cursor_with_an_arrow_above_it(self):
        self.widget.go_to(self.locus.start(), center=True)
        [cursor] = self.cursor_cells()
        self.assertEqual(self.arrows(), [(cursor[0], cursor[1] - 1)])

    def test_cursor_rests_on_the_hidden_column_that_holds_the_target(self):
        self.widget.go_to(self.locus.start(), center=True)
        [cursor] = self.cursor_cells()
        self.assertEqual(self.cells()[cursor]["text"], "\u2026")
        self.widget.show(self.locus.start(), center=True)
        self.assertEqual(self.cursor_cells(), [])
        self.widget.show_cursor()
        [cursor] = self.cursor_cells()
        self.assertEqual(self.cells()[cursor]["text"], "\u2026")

    def test_hide_cursor_keeps_the_cursor_out_until_show_cursor(self):
        self.widget.hide_cursor().go_to(self.locus.start())
        self.assertEqual(self.cursor_cells(), [])
        self.assertEqual(self.arrows(), [])
        self.widget.show_cursor()
        self.assertEqual(len(self.cursor_cells()), 1)

    def test_zooming_in_from_truncated_lands_on_the_target(self):
        self.widget.go_to(self.locus.start(), center=True)
        self.widget.zoom_in().zoom_in()
        [cursor] = self.cursor_cells()
        self.assertEqual(self.row_text(cursor[1]).find(self.needle), cursor[0])

    def test_zooming_out_and_back_keeps_the_target_in_place(self):
        self.widget.show(self.locus.start(), center=True).show_cursor().zoom_in().zoom_in()
        [before] = self.cursor_cells()
        self.widget.zoom_out().zoom_out().zoom_in().zoom_in()
        self.assertEqual(self.cursor_cells(), [before])
        self.assertEqual(self.row_text(before[1]).find(self.needle), before[0])

    def test_zooming_after_a_scroll_keeps_the_node_on_screen(self):
        self.widget.go_to(self.locus.start()).zoom_in().zoom_in().scroll_right()
        rows = range(10)
        before = [self.row_text(y) for y in rows]
        self.widget.zoom_out()
        self.assertTrue(any("\u255f" in self.row_text(y) for y in rows))
        self.widget.zoom_in()
        self.assertEqual([self.row_text(y) for y in rows], before)

    def test_show_highlights_in_one_color_every_time(self):
        self.widget.zoom_in().hide_cursor()
        for start in (self.NEEDLE_START, self.NEEDLE_START + 20):
            [locus] = self.graph.search(
                self.sequence[start : start + 8], sequence_kind="exact"
            )
            self.widget.show(locus)
        backgrounds = {
            cell["bg"]
            for cell in self.widget.frame["cells"]
            if cell.get("bg") not in (None, NODE_BACKGROUND)
        }
        self.assertEqual(backgrounds, {HIGHLIGHT})

    def test_highlight_survives_every_zoom_level(self):
        self.widget.show(self.locus).hide_cursor()
        for step in (
            self.widget.zoom_out,
            self.widget.zoom_out,
            self.widget.zoom_in,
            self.widget.zoom_in,
        ):
            step()
            tinted = [
                cell
                for cell in self.widget.frame["cells"]
                if HIGHLIGHT in (cell.get("bg"), cell.get("fg"))
            ]
            self.assertTrue(tinted)
