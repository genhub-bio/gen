# Visualizing graphs: plot, navigate, show to a user

Signatures were checked against `gen-python/src/python_api/` and the stub
`gen-python/python/gen/gen/__init__.pyi`. `sample.plot()` and `graph.plot()` are documented
there.

Contents: plotting; which widget you get; navigating; showing a plot to a user; mistakes to
avoid.

## Plotting

```python
widget = graph.plot(rows=20, cols=100, align_path=True, show_history=False)
# sample.plot() pages through the sample's graphs.
widget.show(locus, color="cyan", center=False)    # each returns the widget
widget.go_to(locus.start(), center=False)
widget.refresh()
widget.next_page(); widget.prev_page()
widget.zoom_in(); widget.zoom_out()
widget.show_labels(); widget.hide_labels()   # annotation names at normal zoom
widget.scroll_left(); widget.scroll_right(); widget.scroll_up(); widget.scroll_down()
print(widget)   # text rendering
```

Graph plot supports `detail`, `colors`, `align_path`, and `show_history`; sample plot supports
`colors`, `align_path`, and `show_history`. `align_path` (default `True`) lays the current path
out as one straight row with variation branching above and below; the current path is the one
the viewer follows and need not be the reference. `align_path=False` restores the centered
layout. `show_history` (default `False`) adds retired/pruned alternates dimmed and dashed edges;
without it they are not drawn. It is not an operation diff. `detail` accepts `"full"` (complete sequence in each node), `"normal"` (truncated, with
ellipses) and `"minimal"` (filled circles). Outside a notebook, printing a graph or sample
alone (`print(graph)`) shows only an identifier line; use `.plot()` to get the picture.
Set `interactive=False` on either plot method to get an in-memory `PngImage` instead of a
widget. It uses the plot dimensions as its render canvas, then trims blank space. It displays
as the last expression in a notebook cell without writing a file or requiring IPython as a
Gen dependency. For a sample, the static image shows its first graph.

## Which widget you get

Terminals, scripts, and agent REPLs get `TextGraphWidget` (no Jupyter needed). A live Jupyter
kernel with optional dependencies gets `GraphWidget`; a live kernel without the `jupyter`
extra falls back to text with a one-line install hint. `TextGraphWidget` and the Jupyter
`GraphWidget` have the same methods, and every navigation method returns the
widget, so calls chain: `print(widget.show(locus).zoom_in().scroll_right())`. In notebooks use
the interactive widget freely.

## Navigating

- `show(target, color=None, center=False)` navigates and highlights loci and annotations;
  positions and superpositions navigate without highlighting.
- `go_to(target, center=False)` navigates only.
- `refresh()` redraws after graph edits.
- Paging (`next_page`/`prev_page`) switches graphs in a sample; scrolling
  (`scroll_*`) moves within a graph. Do not confuse them.
- `zoom_in()`/`zoom_out()` step the detail level: minimal, normal with annotation names
  hidden, normal with names shown, full sequence.
- Plots open at normal zoom with annotation names hidden: annotations draw as bars, which keeps
  long names from stretching the nodes and gives a good overview. To read the names, call
  `show_labels()` (and `hide_labels()` to go back) or zoom in once. The full-sequence view
  always names every annotation. `show(locus_or_annotation)` highlights the cells of that
  annotation even while names are hidden, so you can find a feature by its highlight and
  add `show_labels()` when you need the names.
- Also on both widgets: `clear_highlights()`, `show_path()`, `hide_path()`,
  `show_track(name)`, `hide_track(name)`, `.tracks`, `hide_all_tracks()`,
  `handle_click(col, row)`, and `save_png(path, cols=None, rows=None)` to write the view to a PNG
  (no Jupyter needed; `gen view <graph> --sample <s> --png out.png` does the same from the CLI).
  Persist annotations through the graph API
  (`graph.add_annotation`, see `design.md`); highlights are ephemeral.

## Showing a plot to a user

When asked to show, plot, or compare graphs outside a live
Jupyter kernel, `print(widget)` and paste the output in a fenced code block. The first frame
printed in a process starts with a `#` orientation header written for you: it explains that
the graph is a population or library of sequences read left to right, with nodes as sequence
fragments and solid edges as the ways they can be joined into paths. Drop that header and
show only the frame. Later frames have no header. Print one widget per graph when comparing,
and label each block with the sample or graph name. Add one or two sentences of your own
interpreting what the graph shows, since the user did not see the header.

### As an image

Use `widget.save_png("plot.png", cols=120, rows=30)` when the user wants a file (a figure for
a document, slide or README), or when you can view or attach images. It needs no Jupyter
dependencies and trims blank space around the graph. Omitted `cols` and `rows` use the widget's
viewport size; larger values show more content without padding the image. It renders the same
view as the widget, so pass `show_history=True` to `plot()` first if the image should include
dimmed alternates and dashed edges. `save_png` returns `None`; finish navigation before calling it.
Use `graph.plot(interactive=False, rows=30, cols=120)` for an image in memory without a file.
From the shell, `gen view <graph> --sample <s> --png out.png` does the same. Keep the text frame
for answers in chat.

## Mistakes to avoid

- Pasting the `#` orientation header into the answer for the user.
- Calling `print(widget)` before `refresh()` after graph edits and showing a stale frame.
- Expecting an interactive canvas outside a live Jupyter kernel.
- Describing `print(graph)` or `print(sample)` as "the plot".
- Pasting a text frame when the user asked for an image or file; use `save_png`.
