# Visualizing graphs: plot, navigate, show to a user

Signatures were checked against `gen-python/src/python_api/` and the stub
`gen-python/python/gen/gen/__init__.pyi`. `sample.plot()` and `graph.plot()` are documented
there.

Contents: plotting; which widget you get; navigating; showing a plot to a user; mistakes to
avoid.

## Plotting

```python
widget = graph.plot(rows=20, cols=100, show_history=False)
# sample.plot() pages through the sample's graphs.
widget.show(locus, color="cyan", center=False)    # each returns the widget
widget.go_to(locus.start(), center=False)
widget.refresh()
widget.next_page(); widget.prev_page()
widget.zoom_in(); widget.zoom_out()
widget.scroll_left(); widget.scroll_right(); widget.scroll_up(); widget.scroll_down()
print(widget)   # text rendering
```

Graph plot supports `detail`, `colors`, and `show_history`; sample plot supports `colors` and
`show_history`. History display includes retired/pruned edges dimmed; it is not an operation
diff. `detail` accepts `"full"` (complete sequence in each node), `"normal"` (truncated, with
ellipses) and `"minimal"` (filled circles). Outside a notebook, printing a graph or sample
alone (`print(graph)`) shows only an identifier line; use `.plot()` to get the picture.

## Which widget you get

Terminals, scripts, and agent REPLs get `TextGraphWidget` (no Jupyter needed). A live Jupyter
kernel with optional dependencies gets `GraphWidget`; a live kernel without the `jupyter`
extra falls back to text with a one-line install hint. `TextGraphWidget` and the Jupyter
`GraphWidget` have the same methods, and every navigation or display method returns the
widget, so calls chain: `print(widget.show(locus).zoom_in().scroll_right())`. In notebooks use
the interactive widget freely.

## Navigating

- `show(target, color=None, center=False)` navigates and highlights loci and annotations;
  positions and superpositions navigate without highlighting.
- `go_to(target, center=False)` navigates only.
- `refresh()` redraws after graph edits.
- Paging (`next_page`/`prev_page`) switches graphs in a sample; scrolling
  (`scroll_*`) moves within a graph. Do not confuse them.
- `zoom_in()`/`zoom_out()` step the detail level.
- Also on both widgets: `clear_highlights()`, `show_path()`, `hide_path()`,
  `show_track(name)`, `hide_track(name)`, `.tracks`, `hide_all_tracks()`,
  `handle_click(col, row)`. Persist annotations through the graph API
  (`graph.add_annotation`, see `design.md`); highlights are ephemeral.

## Showing a plot to a user

`TextWidget` is `TextGraphWidget`. When asked to show, plot, or compare graphs outside a live
Jupyter kernel, `print(widget)` and paste the output in a fenced code block. The first frame
printed in a process starts with a `#` orientation header written for you: it explains that
the graph is a population or library of sequences read left to right, with nodes as sequence
fragments and solid edges as the ways they can be joined into paths. Drop that header and
show only the frame. Later frames have no header. Print one widget per graph when comparing,
and label each block with the sample or graph name. Add one or two sentences of your own
interpreting what the graph shows, since the user did not see the header.

## Mistakes to avoid

- Pasting the `#` orientation header into the answer for the user.
- Calling `print(widget)` before `refresh()` after graph edits and showing a stale frame.
- Expecting an interactive canvas outside a live Jupyter kernel.
- Describing `print(graph)` or `print(sample)` as "the plot".
