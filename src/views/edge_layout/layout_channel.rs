use std::{
    cmp,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
};

use itertools::Itertools;

use super::{EdgeData, LayoutError, NodeData, temp_graph::TempGraph};

#[derive(Clone, Debug)]
struct JogScore {
    pub number_freed: i64,
    pub distance_ranking: Vec<i64>,
    pub jog_length_sum: i64,
}

#[derive(Debug)]
struct GraphBuilder<'a> {
    pub node_positions_by_id: HashMap<u64, (i64, i64)>,
    pub node_ids_by_position: HashMap<(i64, i64), u64>,
    pub node_ports_by_id: HashMap<u64, (i64, i64, i64, i64)>,
    pub edges: Vec<(u64, u64)>,
    pub roles_by_edge: HashMap<(u64, u64), Vec<&'a str>>,
    pub nets_by_edge: HashMap<(u64, u64), u64>,
}

static NODE_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

impl GraphBuilder<'_> {
    fn new() -> Self {
        GraphBuilder {
            node_positions_by_id: HashMap::new(),
            node_ids_by_position: HashMap::new(),
            node_ports_by_id: HashMap::new(),
            edges: vec![],
            roles_by_edge: HashMap::new(),
            nets_by_edge: HashMap::new(),
        }
    }

    fn add_node_at_position(&mut self, x: i64, y: i64) -> u64 {
        if let Some(node_id) = self.node_ids_by_position.get(&(x, y)) {
            *node_id
        } else {
            let node_id = NODE_ID_COUNTER.load(AtomicOrdering::SeqCst) as u64;
            NODE_ID_COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
            self.node_ids_by_position.insert((x, y), node_id);
            self.node_positions_by_id.insert(node_id, (x, y));
            node_id
        }
    }

    fn get_node_position(&self, node_id: u64) -> Option<(i64, i64)> {
        self.node_positions_by_id.get(&node_id).copied()
    }

    fn has_node_at_position(&self, x: i64, y: i64) -> bool {
        self.node_ids_by_position.contains_key(&(x, y))
    }

    fn node_ids(&self) -> Vec<u64> {
        self.node_ids_by_position
            .values()
            .copied()
            .collect::<Vec<u64>>()
    }

    fn transpose(&mut self) {
        for (_node_id, position) in self.node_positions_by_id.iter_mut() {
            let (x, y) = position;
            *position = (*y, *x);
        }

        let mut new_ids_by_position = HashMap::new();
        for (position, node_id) in self.node_ids_by_position.iter() {
            let (x, y) = position;
            new_ids_by_position.insert((*y, *x), *node_id);
        }

        self.node_ids_by_position.clear();
        for (position, node_id) in new_ids_by_position.iter() {
            self.node_ids_by_position.insert(*position, *node_id);
        }

        for (_node_id, port) in self.node_ports_by_id.iter_mut() {
            let (n, e, s, w) = port;
            *port = (*e, *n, *w, *s);
        }
    }

    fn build_graph(&self) -> Result<TempGraph, LayoutError> {
        let mut graph = TempGraph::new();

        let edge_nodes = self
            .edges
            .iter()
            .flat_map(|e| [e.0, e.1])
            .collect::<HashSet<u64>>();

        for node_id in &edge_nodes {
            let position = self.node_positions_by_id.get(node_id);
            if let Some(position) = position {
                let node_data = NodeData {
                    node_id: *node_id,
                    position: *position,
                    node_type: None,
                    ports: None,
                    glyph_index: None,
                    size: (1, 1),
                };
                graph.add_node(*node_id, node_data);
            }
        }

        let mut edge_id_counter = 1;
        for edge in &self.edges {
            graph.add_edge(
                edge_id_counter,
                edge.0,
                edge.1,
                EdgeData {
                    role: Some("Rectilinear".to_string()),
                },
            )?;
            edge_id_counter += 1;
        }

        Ok(graph)
    }
}

// Regarding top/bottom vs. left/right pins: Even though in our application we
// have our graph laid out horizontally (edges run left-to-right), we implement
// the channel routing algorithm assuming edges run vertically. This is closer
// to the literature and just requires transposing the graph at certain
// points. The y-axis points upwards to match the canvas coordinate system (not
// the screen coordinates).
#[derive(Clone, Debug)]
pub struct Router {
    pub bottom_pin_list: Vec<u64>,
    pub top_pin_list: Vec<u64>,
    pub minimum_jog_length: i64,
    pub steady_net_constant: i64,
    pub current_column: i64,
    pub channel_length: i64,
    pub channel_width: i64,
}

impl Router {
    fn next_pin(&self, net: Option<u64>, side: Option<&str>) -> Option<i64> {
        let top;
        let bottom;
        if let Some(net) = net {
            top = self
                .top_pin_list
                .iter()
                .enumerate()
                .filter(|(i, x)| **x == net && *i as i64 > self.current_column)
                .map(|(i, _)| i as i64)
                .min();
            bottom = self
                .bottom_pin_list
                .iter()
                .enumerate()
                .filter(|(i, x)| **x == net && *i as i64 > self.current_column)
                .map(|(i, _)| i as i64)
                .min();
        } else {
            top = self
                .top_pin_list
                .iter()
                .enumerate()
                .filter(|(i, _)| *i as i64 > self.current_column)
                .map(|(i, _)| i as i64)
                .min();
            bottom = self
                .bottom_pin_list
                .iter()
                .enumerate()
                .filter(|(i, _)| *i as i64 > self.current_column)
                .map(|(i, _)| i as i64)
                .min();
        }

        if let Some(side) = side {
            if side == "T" {
                top
            } else if side == "B" {
                bottom
            } else {
                None
            }
        } else if let Some(top) = top {
            if let Some(bottom) = bottom {
                Some(cmp::min(top, bottom))
            } else {
                Some(top)
            }
        } else {
            bottom
        }
    }

    fn classify_net(&self, net: u64) -> &'static str {
        let next_top = self.next_pin(Some(net), Some("T"));
        let next_bottom = self.next_pin(Some(net), Some("B"));

        if let Some(next_top) = next_top {
            if let Some(next_bottom) = next_bottom {
                if next_bottom >= next_top + self.steady_net_constant {
                    "rising"
                } else {
                    "falling"
                }
            } else {
                "rising"
            }
        } else if next_bottom.is_some() {
            "falling"
        } else {
            "steady"
        }
    }

    // Step 1: Make feasible top and bottom connections in minimal manner
    // ------------------------------------------------------------------
    fn add_vertical_wire(
        &self,
        net: u64,
        from_track: i64,
        to_track: i64,
        graph_builder: &mut GraphBuilder,
    ) {
        // Ensure that y1 < y2:
        let y1 = cmp::min(from_track, to_track);
        let y2 = cmp::max(from_track, to_track);
        let node1_id = graph_builder.add_node_at_position(self.current_column, y1);
        let node2_id = graph_builder.add_node_at_position(self.current_column, y2);

        graph_builder.edges.push((node1_id, node2_id));
        graph_builder.nets_by_edge.insert((node1_id, node2_id), net);
        graph_builder
            .roles_by_edge
            .insert((node1_id, node2_id), vec!["Rectilinear"]);
    }

    fn vertical_wiring(&self, graph_builder: &mut GraphBuilder) -> Vec<(i64, i64, u64)> {
        let mut vertical_wires = vec![];

        for (u_id, v_id) in graph_builder.edges.iter() {
            if let Some(net) = graph_builder.nets_by_edge.get(&(*u_id, *v_id)) {
                let u_position = graph_builder.get_node_position(*u_id);
                let v_position = graph_builder.get_node_position(*v_id);

                if let Some((x1, y1)) = u_position
                    && let Some((x2, y2)) = v_position
                    && x1 == x2
                    && x1 == self.current_column
                {
                    vertical_wires.push((cmp::min(y1, y2), cmp::max(y1, y2), *net));
                }
            }
        }

        vertical_wires
    }

    fn tracks(&self, tracks_by_net: &HashMap<u64, HashSet<i64>>) -> HashSet<i64> {
        let mut result = HashSet::new();
        for tracks in tracks_by_net.values() {
            result.extend(tracks);
        }
        result
    }

    fn all_tracks(&self) -> HashSet<i64> {
        (1..self.channel_width + 1).collect()
    }

    fn free_tracks(&self, tracks_by_net: &HashMap<u64, HashSet<i64>>) -> HashSet<i64> {
        self.all_tracks()
            .difference(&self.tracks(tracks_by_net))
            .copied()
            .collect()
    }

    fn connect_pins(
        &self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
        graph_builder: &mut GraphBuilder,
    ) {
        let top_net = self.top_pin_list[self.current_column as usize];
        let bottom_net = self.bottom_pin_list[self.current_column as usize];
        let y1 = 0;
        let y2 = self.channel_width + 1;

        let vertical_wiring = self.vertical_wiring(graph_builder);

        // Use the vertical wires to check if the pins have already been connected
        let bottom_connected = vertical_wiring.iter().any(|(y1, _, _)| *y1 == 0);
        let top_connected = vertical_wiring
            .iter()
            .any(|(_, y2, _)| *y2 == self.channel_width + 1);

        if top_connected && bottom_connected {
            return;
        }

        let top_net_tracks = if let Some(tracks) = tracks_by_net.get(&top_net) {
            tracks
        } else {
            &HashSet::new()
        };

        let next_top_pin = self.next_pin(Some(top_net), None);
        // Special case:
        //     if there are no empty tracks, and net Ti = Bi =/=0 is a net which has connections in this column only,
        //     then run a vertical wire from top to bottom of this column
        if top_net != 0
            && bottom_net != 0
            && top_net == bottom_net
            && self.tracks(tracks_by_net).len() as i64 == self.channel_width
            && top_net_tracks.is_empty()
            && next_top_pin.is_none()
        {
            // Vertical wire from bottom to top
            self.add_vertical_wire(top_net, y1, y2, graph_builder);
            return;
        }

        let free_tracks = self.free_tracks(tracks_by_net);

        // Find the nearest track for the top and/or bottom pins
        let bottom_track = if bottom_net != 0 && !bottom_connected {
            let tracks = tracks_by_net.get(&bottom_net);
            let possible_tracks = if let Some(tracks) = tracks {
                free_tracks.union(tracks).copied().collect()
            } else {
                free_tracks.clone()
            };
            possible_tracks.into_iter().min()
        } else {
            None
        };

        let top_track = if top_net != 0 && !top_connected {
            let tracks = tracks_by_net.get(&top_net);
            let possible_tracks = if let Some(tracks) = tracks {
                free_tracks.union(tracks).copied().collect()
            } else {
                free_tracks.clone()
            };
            possible_tracks.into_iter().max()
        } else {
            None
        };

        // If there is overlap, only keep the shortest vertical wire,
        // the other pin will be connected when the channel is widened.
        if let (Some(bottom_track), Some(top_track)) = (bottom_track, top_track)
            && bottom_net != 0
            && !bottom_connected
            && top_net != 0
            && !top_connected
        {
            // Check if the same net is connecting top and bottom
            if top_net == bottom_net {
                // Same net (T[i] == B[i] != 0): Allow overlap
                if let Some(entry) = tracks_by_net.get_mut(&bottom_net) {
                    entry.insert(bottom_track);
                };
                if let Some(entry) = tracks_by_net.get_mut(&top_net) {
                    entry.insert(top_track);
                };
                self.add_vertical_wire(bottom_net, 0, bottom_track, graph_builder);
                self.add_vertical_wire(top_net, top_track, self.channel_width + 1, graph_builder);
            } else {
                // Different nets:
                if bottom_track < top_track {
                    if let Some(entry) = tracks_by_net.get_mut(&bottom_net) {
                        entry.insert(bottom_track);
                    };
                    if let Some(entry) = tracks_by_net.get_mut(&top_net) {
                        entry.insert(top_track);
                    };
                    self.add_vertical_wire(bottom_net, 0, bottom_track, graph_builder);
                    self.add_vertical_wire(
                        top_net,
                        top_track,
                        self.channel_width + 1,
                        graph_builder,
                    );
                } else {
                    // Overlap, only keep the shortest vertical wire
                    // Compare vertical distances: bottom pin vs top pin
                    if bottom_track < (self.channel_width + 1 - top_track) {
                        if let Some(entry) = tracks_by_net.get_mut(&bottom_net) {
                            entry.insert(bottom_track);
                        };
                        self.add_vertical_wire(bottom_net, 0, bottom_track, graph_builder);
                    } else {
                        if let Some(entry) = tracks_by_net.get_mut(&top_net) {
                            entry.insert(top_track);
                        };
                        self.add_vertical_wire(
                            top_net,
                            top_track,
                            self.channel_width + 1,
                            graph_builder,
                        );
                    }
                }
            }
        } else if let Some(bottom_track) = bottom_track
            && bottom_net != 0
            && !bottom_connected
        {
            if let Some(entry) = tracks_by_net.get_mut(&bottom_net) {
                entry.insert(bottom_track);
            };
            self.add_vertical_wire(bottom_net, 0, bottom_track, graph_builder);
        } else if let Some(top_track) = top_track
            && top_net != 0
            && !top_connected
        {
            if let Some(entry) = tracks_by_net.get_mut(&top_net) {
                entry.insert(top_track);
            };
            self.add_vertical_wire(top_net, top_track, self.channel_width + 1, graph_builder);
        }
    }

    fn split_nets(tracks_by_net: &HashMap<u64, HashSet<i64>>) -> Vec<u64> {
        let mut result = HashSet::new();
        for (net, tracks) in tracks_by_net.iter() {
            if tracks.len() > 1 {
                result.insert(net);
            }
        }

        result.into_iter().copied().collect::<Vec<u64>>()
    }

    // Step 2: Free as many tracks as possible by collapsing split nets
    // ----------------------------------------------------------------
    fn generate_jog_patterns(
        &self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
    ) -> Vec<Vec<Vec<(i64, i64)>>> {
        // Generate all possible jog patterns for the current column
        // Returns a pattern as a list of jogs, grouped by the net they belong to:
        // [((track1, track2), (track3, track4), ...), ((track5, track6), (track7, track8), ...), ...]
        // (This also includes empty groups to keep the distinction between nets)

        let mut jogs_partitioned_by_net: Vec<Vec<(i64, i64)>> = vec![];
        let split_nets = Self::split_nets(tracks_by_net);
        for net in split_nets.iter().sorted() {
            // Generate all possible jogs for this net (non-overlapping)
            let track_set = tracks_by_net.get(net);
            if let Some(track_set) = track_set {
                let track_list = track_set.iter().copied().sorted().collect::<Vec<i64>>();
                jogs_partitioned_by_net.push(track_list.iter().copied().tuple_windows().collect());
            }
        }

        // To build the patterns we take the cartesian product of the power sets of each net
        let mut jog_powersets: Vec<Vec<Vec<(i64, i64)>>> = vec![];

        for net_jogs in jogs_partitioned_by_net {
            let powerset: Vec<Vec<(i64, i64)>> = net_jogs.iter().cloned().powerset().collect();
            jog_powersets.push(powerset.clone());
        }

        jog_powersets
            .iter()
            .map(|v| v.iter())
            .multi_cartesian_product()
            .map(|v| {
                v.into_iter()
                    .cloned()
                    .collect::<Vec<std::vec::Vec<(i64, i64)>>>()
            })
            .filter(|pattern| Self::validate_pattern(pattern))
            .collect()
    }

    fn validate_pattern(pattern: &[Vec<(i64, i64)>]) -> bool {
        // Pattern is a list of jogs, grouped by net:
        // [((track1, track2), (track3, track4), ...), ((track5, track6), (track7, track8), ...), ...]
        // Check if the jogs in the pattern are valid by testing for overlaps
        // between the jogs of different nets. This is a two-step combination:
        // 1) each net is checked against all other nets
        // 2) all jogs from one net are checked against all jogs from the other net
        // Returns True if valid, False otherwise

        for jog_pair in pattern.iter().combinations(2) {
            let net1_jogs = jog_pair[0];
            let net2_jogs = jog_pair[1];
            for j1 in net1_jogs {
                for j2 in net2_jogs {
                    let (low1, high1) = j1;
                    let (low2, high2) = j2;
                    assert!(low1 < high1 && low2 < high2);
                    if !(high1 < low2 || high2 < low1) {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn contiguous(&self, jogs: Vec<(i64, i64)>) -> bool {
        // Tests if the given pairs of segments are contiguous.
        // Each pair is a tuple of (start, stop), sort them by start
        let new_jogs: Vec<(i64, i64)> = jogs
            .clone()
            .iter()
            .sorted_by_key(|jog| jog.0)
            .copied()
            .collect();
        for (pair1, pair2) in new_jogs.iter().tuple_windows() {
            if pair1.1 != pair2.0 {
                return false;
            }
        }

        true
    }

    fn evaluate_jogs(
        &self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
        pattern: &Vec<Vec<(i64, i64)>>,
    ) -> JogScore {
        // pattern is a list of lists of tuples (track1,track2) grouped by net
        // Returns a score as 3 values:
        // 1. Number of tracks freed
        // 2. Outermost split net distance from edge
        // 3. Sum of jog lengths

        // 1) Number of new empty tracks created by the jogs (higher is better)
        // From the paper: "a pattern [of jogs] will free up one track for every jog it contains,
        // plus one additional track for every net it finishes"
        let all_jogs: Vec<(i64, i64)> = pattern.iter().flatten().copied().collect();
        let mut number_freed = all_jogs.len() as i64;

        // The only nets we can finish are the split nets that are still being routed, but don't have an upcoming pin
        let mut almost_finished_nets = vec![];
        for (net, tracks) in tracks_by_net.iter() {
            if (self.next_pin(Some(*net), None).is_none()) && tracks.len() > 1 {
                almost_finished_nets.push(*net);
            }
        }

        for group in pattern {
            if group.is_empty() {
                continue;
            }

            // If the jogs for that net are not contiguous, it won't fully close out the net
            // so we don't count it.
            if !self.contiguous(group.clone()) {
                continue;
            }

            // Test if there's a net for which we've just freed all the tracks
            // and that there are no pins coming up anymore.
            for net in &almost_finished_nets {
                let tracks = tracks_by_net.get(net);
                if let Some(tracks) = tracks {
                    let mut group_tracks = HashSet::new();
                    for jog in group {
                        group_tracks.insert(jog.0);
                        group_tracks.insert(jog.1);
                    }
                    if group_tracks.is_subset(tracks) {
                        number_freed += 1;
                        break;
                    }
                }
            }
        }

        // 2) Maximize the distance of the outermost split net from the edge
        // Find all split nets that would not be joined by the jogs
        // Then find the outermost track of each of those nets
        // Then take the minimum distance of those outermost tracks from the edge
        let mut net_distances = vec![];
        let all_tracks: HashSet<i64> = all_jogs
            .clone()
            .into_iter()
            .flat_map(|pair| vec![pair.0, pair.1])
            .collect();
        for net in Self::split_nets(tracks_by_net) {
            let tracks = tracks_by_net.get(&net);
            if let Some(tracks) = tracks {
                let dangling_tracks: HashSet<i64> =
                    tracks.difference(&all_tracks).copied().collect();
                if dangling_tracks.is_empty() {
                    continue;
                }
                let min_dangling_track = dangling_tracks.iter().min().unwrap();
                let distance_from_bottom = min_dangling_track - 1;
                let max_dangling_track = dangling_tracks.iter().max().unwrap();
                let distance_from_top = self.channel_width - max_dangling_track;
                if distance_from_bottom < distance_from_top {
                    net_distances.push(distance_from_bottom);
                } else {
                    net_distances.push(distance_from_top);
                }
            }
        }

        // Save a sorted list so that we can also compare the second net etc.
        let distance_ranking = net_distances.clone().iter().sorted().copied().collect();

        // 3) Minimize the total length of the jogs
        let mut jog_length_sum = 0;
        for (y1, y2) in &all_jogs {
            jog_length_sum += y2 - y1;
        }

        JogScore {
            number_freed,
            distance_ranking,
            jog_length_sum,
        }
    }

    fn compare_scores(score1: &JogScore, score2: &JogScore) -> bool {
        // Find the best pattern with multiple tiebreakers
        // returns True if score1 is better than score2

        // Maximize the number of tracks freed
        if score1.number_freed != score2.number_freed {
            return score1.number_freed > score2.number_freed;
        }

        // Maximize the distance of the outermost split net from the edge
        // If the distance is the same, then compare the second outermost net etc.

        // TODO: Can these lengths differ?
        assert!(score1.distance_ranking.len() == score2.distance_ranking.len());
        for (d1, d2) in score1
            .distance_ranking
            .clone()
            .into_iter()
            .zip(score2.distance_ranking.clone())
        {
            if d1 != d2 {
                return d1 > d2;
            }
        }

        // Maximize the total length of the jogs
        score1.jog_length_sum > score2.jog_length_sum
    }

    fn overlaps(pairs: Vec<(i64, i64)>) -> bool {
        // Tests for overlaps between any pairs in a list of pairs.
        if pairs.len() == 1 {
            return false;
        }

        // Check for overlaps
        for (pair1, pair2) in pairs.iter().sorted_by_key(|pair| pair.0).tuple_windows() {
            let stop1 = pair1.1;
            let start2 = pair2.0;
            // An overlap occurs if the next segment starts before the previous one ends
            if stop1 >= start2 {
                // somewhat controversial
                return true;
            }
        }

        false
    }

    fn collapse_split_nets(
        &mut self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
        graph_builder: &mut GraphBuilder,
    ) {
        // Finds an optimal pattern of jogs between tracks holding split nets
        if Self::split_nets(tracks_by_net).is_empty() {
            return;
        }

        // Generate all legal jog combinations for the current column
        let jog_patterns = self.generate_jog_patterns(tracks_by_net);
        if jog_patterns.is_empty() {
            return;
        }

        // Filter out any jog pattern that would overlap with an existing vertical
        // wire from a DIFFERENT net in this column.
        let mut filtered_patterns: Vec<Vec<Vec<(i64, i64)>>> = vec![];
        let mut existing_verticals = vec![]; // tuples: (y_low, y_high, net)

        let split_nets = Self::split_nets(tracks_by_net);

        for (u_id, v_id) in graph_builder.edges.iter() {
            let net = graph_builder.nets_by_edge.get(&(*u_id, *v_id)).unwrap();
            let u_pos = graph_builder.get_node_position(*u_id);
            let v_pos = graph_builder.get_node_position(*v_id);

            if let Some((x1, y1)) = u_pos
                && let Some((x2, y2)) = v_pos
                && x1 == x2
                && x2 == self.current_column
            {
                existing_verticals.push((cmp::min(y1, y2), cmp::max(y1, y2), net));
            }
        }

        for pattern in &jog_patterns {
            let mut valid = true;
            for (idx, group) in pattern.iter().enumerate() {
                let net = &split_nets[idx]; // group corresponds to this net
                for (y1, y2) in group {
                    let jog_pair = (cmp::min(y1, y2), cmp::max(y1, y2));
                    for (v_y1, v_y2, v_net) in &existing_verticals {
                        if *v_net != net
                            && Router::overlaps(vec![(*jog_pair.0, *jog_pair.1), (*v_y1, *v_y2)])
                        {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        break;
                    }
                }
                if !valid {
                    break;
                }
            }
            if valid {
                filtered_patterns.push((*pattern.clone()).to_vec());
            }
        }

        let jog_patterns = filtered_patterns;
        if jog_patterns.is_empty() {
            return;
        }

        // Test all combinations of jogs to find the pattern that creates the most empty tracks
        let mut best_pattern: Option<Vec<Vec<(i64, i64)>>> = None;

        // The score is a tuple of 3 values (tracks freed and tiebreakers)
        let mut best_score = JogScore {
            number_freed: 0,
            distance_ranking: vec![],
            jog_length_sum: self.channel_width,
        }; // This will always lose

        for combo in &jog_patterns {
            let score = self.evaluate_jogs(tracks_by_net, combo);
            if best_score.distance_ranking.is_empty() {
                // initial value
                best_score = score;
                best_pattern = Some((*combo.clone()).to_vec());
            } else if Self::compare_scores(&score, &best_score) {
                best_score = score;
                best_pattern = Some((*combo.clone()).to_vec());
            }
        }

        let split_nets = Self::split_nets(tracks_by_net);

        if let Some(best_pattern) = best_pattern {
            // The groups are still in the same order as the split nets (which stays sorted)
            for (i, group) in best_pattern.iter().enumerate() {
                let net = split_nets[i];

                // Add a vertical segment to the net and free up one of the tracks
                for (y1, y2) in group {
                    self.add_vertical_wire(net, *y1, *y2, graph_builder);

                    if let Some(tracks) = tracks_by_net.get_mut(&net) {
                        tracks.remove(y1);
                    }
                }
                // If the net is closed, y2 will be removed in a later step
            }
        } else {
            println!("No valid patterns found");
        }
    }

    // Step 3: Add jogs to reduce the range of split nets
    // --------------------------------------------------
    fn occupied_tracks(&self, tracks_by_net: &HashMap<u64, HashSet<i64>>) -> HashSet<i64> {
        let mut result = HashSet::new();
        for tracks in tracks_by_net.values() {
            result.extend(tracks);
        }
        result
    }

    fn scout(
        &self,
        tracks_by_net: &HashMap<u64, HashSet<i64>>,
        net: u64,
        track: i64,
        goal: i64,
        graph_builder: &mut GraphBuilder,
    ) -> i64 {
        // Find the closest reachable track in the direction of the goal (another track on the same net, or an empty track).
        // Assumes that there are not other tracks of the same net in the way
        // Returns the position of that track number if successful, or the original track if not.

        // Rust is very picky about ranges and reversed ranges.  They are not
        // the same type.  Clippy doesn't understand that they need to be
        // converted to a vector in order for the compiler to be happy.
        #[allow(clippy::useless_conversion)]
        let tracks: Vec<_> = match goal.cmp(&track) {
            Ordering::Greater => (track + 1..goal + 1).into_iter().collect(),
            Ordering::Less => (track - 1..goal - 1).rev().into_iter().collect(),
            Ordering::Equal => {
                return track;
            }
        };

        // We scan the tracks in the direction of a conductor on the same net,
        // and keep a marker of the last reachable track.
        let mut marker = track;
        let vertical_wiring = self.vertical_wiring(graph_builder);
        let occupied_tracks = self.occupied_tracks(tracks_by_net);
        let net_tracks = tracks_by_net.get(&net);
        for i in tracks.iter().copied() {
            // If the vertical layer is occupied, we have to stop the search.
            for (y1, y2, _) in &vertical_wiring {
                if *cmp::min(y1, y2) <= i && i <= *cmp::max(y1, y2) {
                    break;
                }

                // If the horizontal layer is occupied we can jump over it but not land there
                if let Some(net_tracks) = net_tracks
                    && occupied_tracks.iter().contains(&i)
                    && !net_tracks.contains(&i)
                {
                    continue;
                }

                // If we made it this far, we can record the index of this iteration in the marker variable
                marker = i;
            }
        }

        marker
    }

    fn jog(
        &self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
        net: u64,
        track: i64,
        goal: i64,
        graph_builder: &mut GraphBuilder,
    ) -> i64 {
        // Jog the net from track to as close as possible to goal
        let destination = self.scout(tracks_by_net, net, track, goal, graph_builder);

        if destination != track {
            if let Some(entry) = tracks_by_net.get_mut(&net) {
                entry.remove(&track);
            };
            if let Some(entry) = tracks_by_net.get_mut(&net) {
                entry.insert(destination);
            };
            self.add_vertical_wire(net, track, destination, graph_builder);
        }

        destination
    }

    fn compress_split_net(
        &self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
        net: u64,
        graph_builder: &mut GraphBuilder,
    ) {
        // For split nets that weren't collapsed, try to move the tracks closer to each other:
        //  - jog the lowest track up as far as possible
        //  - jog the highest track down as far as possible
        // To find the correct open spot we process the column cell by cell, on both layers

        let track_set = tracks_by_net.get(&net).unwrap();

        let tracks: Vec<i64> = track_set.iter().copied().sorted().collect::<Vec<i64>>();

        // 1) Attempt to jog the lowest track up as far as possible
        let low_track = tracks[0];
        let goal = tracks[1];
        let low_marker = self.scout(tracks_by_net, net, low_track, goal, graph_builder);

        // 2) Attempt to jog the highest track down as far as possible
        let high_track = tracks[tracks.len() - 1];
        let goal = tracks[tracks.len() - 2];
        let high_marker = self.scout(tracks_by_net, net, high_track, goal, graph_builder);

        // 3) Actually move the tracks if the jog is long enough. Do this after the two attempts
        // above so that we don't invalidate the markers by moving the tracks.
        // High track
        if (high_marker - high_track).abs() >= self.minimum_jog_length {
            self.jog(tracks_by_net, net, high_track, high_marker, graph_builder);
        }

        // Low track
        if (low_marker - low_track).abs() >= self.minimum_jog_length {
            self.jog(tracks_by_net, net, low_track, low_marker, graph_builder);
        }
    }

    // Step 4: Add jogs to raise rising nets and lower falling nets
    // ------------------------------------------------------------
    fn push_unsplit_nets(
        &mut self,
        tracks_by_net: &HashMap<u64, HashSet<i64>>,
        graph_builder: &mut GraphBuilder,
    ) -> Vec<(i64, u64, i64, i64)> {
        let x = self.current_column as usize;

        // We look specifically for nets that are not split and have a pin coming up
        let mut upcoming_pin_list = if x >= self.top_pin_list.len() {
            vec![]
        } else {
            self.top_pin_list[x..].to_vec()
        };
        if x < self.bottom_pin_list.len() {
            upcoming_pin_list.extend(&self.bottom_pin_list[x..]);
        }

        let upcoming_pins = upcoming_pin_list.iter().copied().collect::<HashSet<u64>>();
        let nets_to_jog: Vec<u64> = upcoming_pins
            .iter()
            .filter(|net| {
                **net != 0
                    && tracks_by_net.contains_key(*net)
                    && tracks_by_net.get(*net).unwrap().len() == 1
            })
            .copied()
            .collect();

        // Determine where to push the nets to
        let mut track_distances = vec![];
        for net in nets_to_jog {
            let tracks = tracks_by_net.get(&net);
            if let Some(tracks) = tracks {
                let track = tracks.iter().next();
                if let Some(track) = track {
                    let goal;
                    let classification = self.classify_net(net);
                    if classification == "rising" {
                        goal = self.channel_width;
                    } else if classification == "falling" {
                        goal = 1;
                    } else {
                        continue;
                    }

                    // Record the achievable distance to the goal track
                    let destination = self.scout(tracks_by_net, net, *track, goal, graph_builder);
                    let distance = (track - destination).abs();
                    if distance >= self.minimum_jog_length {
                        track_distances.push((distance, net, *track, goal));
                    }
                } else {
                    println!("No tracks found for net: {net}");
                }
            }
        }

        track_distances
    }

    fn update_with_track_distances(
        &self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
        graph_builder: &mut GraphBuilder,
        track_distances: &[(i64, u64, i64, i64)],
    ) {
        // Execute longer jogs first
        for (_, net, track, goal) in track_distances.iter().sorted_by_key(|distance| -distance.0) {
            self.jog(tracks_by_net, *net, *track, *goal, graph_builder);
        }
    }

    // Step 5: Widen channel if needed to make previously not feasible top or bottom connections
    // -----------------------------------------------------------------------------------------
    fn widen_channel(
        &mut self,
        tracks_by_net: &mut HashMap<u64, HashSet<i64>>,
        from_side: Option<&str>,
        graph_builder: &mut GraphBuilder,
    ) -> Result<(), LayoutError> {
        // Inserts a new track which must be:
        // (a) reachable from the top or bottom
        // (b) as close as possible to the middle of the channel
        // If the track x is selected, then the old tracks x, x+1, ... will be moved up to x+1, x+2, ...
        // Note: we are assuming that this function is only called when there is no space left on the channel

        let mut mid_track = (self.channel_width / 2) + 1; // +1 because tracks are indexed from 1 to channel_width
        if self.channel_width % 2 == 1 {
            mid_track += 1;
        }

        // Find a position for the new track that is as close to the middle as possible,
        // and that is accessible from the pins without violating a vertical constraint.
        let current_vertical_wires = self.vertical_wiring(graph_builder); // Call function once
        let min_start;
        let max_end;

        if current_vertical_wires.is_empty() {
            min_start = 1;
            max_end = self.channel_width; // Before widening, tracks go up to channel_width
        } else {
            min_start = current_vertical_wires
                .iter()
                .min_by(|wire1, wire2| wire1.0.cmp(&wire2.0))
                .unwrap()
                .0; // y1
            max_end = current_vertical_wires
                .iter()
                .max_by(|wire1, wire2| wire1.1.cmp(&wire2.1))
                .unwrap()
                .1; // y2
        }

        let new_track_candidate_bottom = cmp::min(min_start, mid_track);
        let new_track_candidate_top = cmp::max(max_end + 1, mid_track);

        let mut new_track;
        if let Some(from_side) = from_side {
            if from_side == "B" {
                // Moving upwards from the bottom: the start of the first vertical wire,
                // or the middle, whichever comes first. New track is inserted AT this position.
                new_track = new_track_candidate_bottom;
            } else if from_side == "T" {
                // Moving downwards from the top: the end of the last vertical wire + 1,
                // or the middle, whichever comes first. New track is inserted AT this position.
                new_track = new_track_candidate_top;
            } else {
                return Err(LayoutError::InvalidSide(from_side.to_string()));
            }
        } else {
            new_track = mid_track;
        }

        // Ensure new_track is within reasonable bounds (1 to channel_width + 1)
        // If inserting at new_track, all tracks >= new_track shift up.
        // new_track is 1-indexed.
        new_track = cmp::max(1, cmp::min(new_track, self.channel_width + 1));

        self.channel_width += 1;

        // Update the active assignments for all tracks above the new track,
        // starting from the top down so we don't overwrite any existing assignments
        for (_net, old_tracks) in tracks_by_net.iter_mut() {
            let mut new_tracks = HashSet::new();
            for old_track in old_tracks.iter() {
                if *old_track >= new_track {
                    new_tracks.insert(*old_track + 1);
                } else {
                    new_tracks.insert(*old_track);
                }
            }
            *old_tracks = new_tracks;
        }

        // Update the graph by moving up any nodes that are now at or above the new track's position.
        // Also update the pos_to_id mapping.
        let binding = graph_builder.node_positions_by_id.clone();
        let nodes_to_update = binding.keys();

        for node_id in nodes_to_update {
            let position = graph_builder.get_node_position(*node_id);
            if let Some(position) = position {
                let (x, y) = position;
                if y >= new_track {
                    let new_y = y + 1;
                    if let Some(entry) = graph_builder.node_positions_by_id.get_mut(node_id) {
                        *entry = (x, new_y);
                    };
                    // Old coord mapping will be removed when rebuilding updated_pos_to_id
                }
                // else: node coordinates remain unchanged
            }
        }

        // Rebuild pos_to_id from the updated graph node attributes
        graph_builder.node_ids_by_position.clear();
        for (node_id, position) in graph_builder.node_positions_by_id.iter() {
            graph_builder
                .node_ids_by_position
                .insert(*position, *node_id);
        }

        Ok(())
    }

    fn extend_nets(
        &mut self,
        tracks_by_net: &HashMap<u64, HashSet<i64>>,
        graph_builder: &mut GraphBuilder,
    ) {
        // Only extend nets that either are split or have a pin coming up
        for (net, tracks) in tracks_by_net.clone().iter_mut() {
            if tracks.len() == 1 && self.next_pin(Some(*net), None).is_none() {
                // Clear the Y dict entry for this net
                tracks.clear();
            } else {
                // Update the graph for each track
                for track in tracks.iter() {
                    let node1_id = graph_builder.add_node_at_position(self.current_column, *track);
                    let node2_id =
                        graph_builder.add_node_at_position(self.current_column + 1, *track);
                    graph_builder.edges.push((node1_id, node2_id));
                    graph_builder
                        .nets_by_edge
                        .insert((node1_id, node2_id), *net);
                    graph_builder
                        .roles_by_edge
                        .insert((node1_id, node2_id), vec!["Rectilinear"]);
                }
            }
        }

        // Update the channel length if needed
        self.channel_length = cmp::max(self.channel_length, self.current_column + 1);
    }

    fn _add_border_spacing(&self, edge_spacing: i64, graph_builder: &mut GraphBuilder) {
        // Increase the space between nodes at the minimum and maximum y coordinates
        // and all other nodes by a set distance.
        //
        // Parameters:
        //     edge_spacing (float): Distance to add between edge nodes and other nodes

        if graph_builder.node_ids().is_empty() {
            return;
        }

        // Single pass: update positions knowing min_x=0 and max_x=channel_width
        graph_builder.node_ids_by_position.clear();
        let mut new_node_positions_by_id = HashMap::new();
        for (node_id, position) in graph_builder.node_positions_by_id.iter_mut() {
            let (x, y) = position;
            let mut new_y = *y;
            if *y > 0 {
                new_y += edge_spacing;
            }

            if *y == self.channel_width + 1 {
                new_y += edge_spacing;
            }

            new_node_positions_by_id.insert(*node_id, (*x, new_y));
            graph_builder
                .node_ids_by_position
                .insert((*x, new_y), *node_id);
        }

        graph_builder.node_positions_by_id.clear();
        for (node_id, position) in new_node_positions_by_id.iter() {
            graph_builder
                .node_positions_by_id
                .insert(*node_id, *position);
        }
    }

    fn finished(
        &self,
        tracks_by_net: &HashMap<u64, HashSet<i64>>,
        graph_builder: &mut GraphBuilder,
    ) -> bool {
        self.next_pin(None, None).is_none()
            && self.occupied_tracks(tracks_by_net).is_empty()
            && self.pins(graph_builder) == (None, None)
    }

    fn pins(&self, graph_builder: &mut GraphBuilder) -> (Option<u64>, Option<u64>) {
        // Get the nets of any unrouted pins in the current column
        // Returns a tuple (top, bottom) where top and bottom may be int or None
        let x = self.current_column;
        if x >= self.channel_length {
            return (None, None);
        }
        let y_t = self.channel_width + 1;
        let y_b = 0;

        let top_net = self.top_pin_list[x as usize];
        let bottom_net = self.bottom_pin_list[x as usize];

        let top = if !graph_builder.has_node_at_position(x, y_t) {
            Some(top_net)
        } else {
            None
        };

        let bottom = if !graph_builder.has_node_at_position(x, y_b) {
            Some(bottom_net)
        } else {
            None
        };

        (top, bottom)
    }

    /*
    Implements the rectilinear channel routing algorithm described in:
    "A 'greedy' channel router" by Rivest and Fiduccia (1983)"


    Returns:
     */
    pub fn route(&mut self) -> Result<TempGraph, LayoutError> {
        // Route the nets, returns a graph representing the routed nets, with
        // nodes at grid points and edges corresponding to horizontal and
        // vertical wire segments.

        // The algorithm will dynamically extend the channel as needed,
        // but we don't want it to extend indefinitely. This is in case
        // the channel is blocked by a net that cannot be routed.
        let max_length = self.channel_length * 3 / 2;

        let graph_builder = &mut GraphBuilder::new();

        let mut tracks_by_net: HashMap<u64, HashSet<i64>> = HashMap::new();
        for bottom_pin in &self.bottom_pin_list {
            tracks_by_net.insert(*bottom_pin, HashSet::new());
        }
        for top_pin in &self.top_pin_list {
            tracks_by_net.insert(*top_pin, HashSet::new());
        }

        while !self.finished(&tracks_by_net, graph_builder) {
            let x = self.current_column;

            // 1) Connect the pins
            if x < self.channel_length {
                self.connect_pins(&mut tracks_by_net, graph_builder)
            }

            // 2) Collapse split nets to free up tracks
            self.collapse_split_nets(&mut tracks_by_net, graph_builder);

            // 3) Compress remaining split nets to narrow their range
            for net in Self::split_nets(&tracks_by_net) {
                self.compress_split_net(&mut tracks_by_net, net, graph_builder);
            }

            // 4) Add jogs to raise rising nets and lower falling nets
            // We look specifically for nets that are not split and have a pin coming up
            let track_distances = self.push_unsplit_nets(&tracks_by_net, graph_builder);
            self.update_with_track_distances(&mut tracks_by_net, graph_builder, &track_distances);

            // 5) Widen the channel if we were not able to route pins earlier because of space constraints
            let (top_net, bottom_net) = self.pins(graph_builder);
            if top_net.is_some() {
                self.widen_channel(&mut tracks_by_net, Some("T"), graph_builder)?;
                self.connect_pins(&mut tracks_by_net, graph_builder);
            }

            if bottom_net.is_some() {
                let _ = self.widen_channel(&mut tracks_by_net, Some("B"), graph_builder);
                self.connect_pins(&mut tracks_by_net, graph_builder);
            }

            // 6) Extend nets to the next column and advance the column pointer
            self.extend_nets(&tracks_by_net, graph_builder);
            self.current_column += 1;

            // Failsafe: if we keep extending the channel without making progress,
            // we'll stop anyway
            if self.channel_length >= max_length {
                break;
            }
        }

        // Add a unit of spacing at the edges of the channel
        self._add_border_spacing(1, graph_builder);

        // Transpose the graph so we have a horizontal layout again
        graph_builder.transpose();

        graph_builder.build_graph()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_tracks() {
        let test_router = Router {
            bottom_pin_list: vec![1, 2, 3],
            top_pin_list: vec![3, 2, 1],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 5,
        };

        assert_eq!((1..6).collect::<HashSet<i64>>(), test_router.all_tracks());
    }

    #[test]
    fn test_occupied_tracks() {
        let test_router = Router {
            bottom_pin_list: vec![1, 2, 3],
            top_pin_list: vec![3, 2, 1],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 5,
        };

        assert_eq!((1..6).collect::<HashSet<i64>>(), test_router.all_tracks());
    }

    #[test]
    fn test_split_nets() {
        let tracks_by_net = HashMap::from([
            (1, HashSet::from([2, 3])),
            (2, HashSet::from([4])),
            (3, HashSet::from([5, 6, 7])),
        ]);
        assert_eq!(
            HashSet::from([1, 3]),
            Router::split_nets(&tracks_by_net)
                .into_iter()
                .collect::<HashSet<u64>>()
        );
    }

    #[test]
    fn test_vertical_wiring() {
        let test_router = Router {
            bottom_pin_list: vec![1, 2, 3],
            top_pin_list: vec![3, 2, 1],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 5,
        };

        let mut test_graph_builder1 = GraphBuilder::new();
        assert_eq!(
            Vec::<(i64, i64, u64)>::new(),
            test_router.vertical_wiring(&mut test_graph_builder1)
        );

        // Add a vertical edge (manually)
        let mut test_graph_builder2 = GraphBuilder::new();
        let node_id1 = test_graph_builder2.add_node_at_position(0, 1);
        let node_id2 = test_graph_builder2.add_node_at_position(0, 2);
        test_graph_builder2.edges.push((node_id1, node_id2));
        test_graph_builder2
            .nets_by_edge
            .insert((node_id1, node_id2), 1);
        // net, track, goal
        assert_eq!(
            vec![(1, 2, 1)],
            test_router.vertical_wiring(&mut test_graph_builder2)
        );

        // # Add a vertical wire
        let test_router2 = Router {
            bottom_pin_list: vec![1, 2, 3],
            top_pin_list: vec![3, 2, 1],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 5,
        };
        let mut test_graph_builder3 = GraphBuilder::new();
        test_router2.add_vertical_wire(1, 1, 2, &mut test_graph_builder3);
        assert_eq!(
            vec![(1, 2, 1)],
            test_router.vertical_wiring(&mut test_graph_builder2)
        );
    }

    #[test]
    fn test_next_pin() {
        // Setup: top and bottom pin lists with multiple pins for net 1
        let mut test_router = Router {
            bottom_pin_list: vec![0, 1, 0, 1],
            top_pin_list: vec![1, 0, 1, 0],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 5,
        };
        // Next pin for net 1, bottom side
        assert_eq!(test_router.next_pin(Some(1), Some("B")), Some(1));
        // Advance column and check again
        test_router.current_column = 2;
        assert_eq!(test_router.next_pin(Some(1), None), Some(3));
        test_router.current_column = 3;
        assert_eq!(test_router.next_pin(Some(1), None), None);
    }

    #[test]
    fn test_connect_pins() {
        // Case 1: Only top pin present, should assign to a free track and add vertical wire
        let mut test_router = Router {
            bottom_pin_list: vec![0],
            top_pin_list: vec![1],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 3,
        };

        let mut test_graph_builder = GraphBuilder::new();

        let mut tracks_by_net = HashMap::from([(1, HashSet::new())]);
        test_router.connect_pins(&mut tracks_by_net, &mut test_graph_builder);

        // Net 1 should occupy a track
        let net1_result = tracks_by_net.get(&1).unwrap();
        assert_eq!(1, net1_result.len());
        // Verify vertical wire to top boundary
        let top_boundary_y = test_router.channel_width + 1;
        let mut found_top_wire = false;
        for (u_id, v_id) in &test_graph_builder.edges {
            let edge_net = test_graph_builder
                .nets_by_edge
                .get(&(*u_id, *v_id))
                .unwrap();
            if *edge_net == 1 {
                let u_pos = test_graph_builder.get_node_position(*u_id);
                let v_pos = test_graph_builder.get_node_position(*v_id);
                if let Some(u_pos) = u_pos
                    && let Some(v_pos) = v_pos
                    && (u_pos.0 == 0 && v_pos.0 == 0 && u_pos.1 == top_boundary_y
                        || v_pos.1 == top_boundary_y)
                {
                    found_top_wire = true;
                    break;
                }
            }
        }

        assert!(found_top_wire);

        // Case 2: Only bottom pin present, should assign to a free track and add vertical wire
        test_router = Router {
            bottom_pin_list: vec![2],
            top_pin_list: vec![0],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 3,
        };
        test_graph_builder = GraphBuilder::new();
        tracks_by_net = HashMap::from([(0, HashSet::new()), (2, HashSet::new())]);
        test_router.connect_pins(&mut tracks_by_net, &mut test_graph_builder);

        let net2_result = tracks_by_net.get(&2).unwrap();
        assert_eq!(1, net2_result.len());

        let mut found = false;
        for (u_id, v_id) in &test_graph_builder.edges {
            let u_pos = test_graph_builder.get_node_position(*u_id);
            let v_pos = test_graph_builder.get_node_position(*v_id);
            if let Some(u_pos) = u_pos
                && let Some(v_pos) = v_pos
                && (u_pos.0 == 0 && v_pos.0 == 0 && u_pos.1 == 0 || v_pos.1 == 0)
            {
                found = true;
                break;
            }
        }
        assert!(found);

        // Case 3: Both pins present, both nets different, should assign both
        test_router = Router {
            bottom_pin_list: vec![2],
            top_pin_list: vec![1],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 3,
        };
        test_graph_builder = GraphBuilder::new();
        tracks_by_net = HashMap::from([(1, HashSet::new()), (2, HashSet::new())]);
        test_router.connect_pins(&mut tracks_by_net, &mut test_graph_builder);

        let net1_result = tracks_by_net.get(&1).unwrap();
        assert_eq!(1, net1_result.len());
        let net2_result = tracks_by_net.get(&2).unwrap();
        assert_eq!(1, net2_result.len());

        // Case 4: Both pins present, same net, all tracks occupied, should add vertical wire from top to bottom
        test_router = Router {
            bottom_pin_list: vec![3],
            top_pin_list: vec![3],
            minimum_jog_length: 10,
            steady_net_constant: 5,
            current_column: 0,
            channel_length: 5,
            channel_width: 0,
        };
        test_graph_builder = GraphBuilder::new();
        tracks_by_net = HashMap::from([(3, HashSet::new())]);
        test_router.connect_pins(&mut tracks_by_net, &mut test_graph_builder);

        // Should have a vertical wire from 0 to channel_width+1
        let mut found = false;
        for (u_id, v_id) in &test_graph_builder.edges {
            let u_pos = test_graph_builder.get_node_position(*u_id);
            let v_pos = test_graph_builder.get_node_position(*v_id);
            if let Some(u_pos) = u_pos
                && let Some(v_pos) = v_pos
                && (u_pos.1 == 0 && v_pos.1 == test_router.channel_width + 1
                    || v_pos.1 == 0 && u_pos.1 == test_router.channel_width + 1)
            {
                found = true;
                break;
            }
        }
        assert!(found);
    }

    #[test]
    fn test_route() {
        let pin_list_pairs = vec![
            (vec![1, 0, 0], vec![0, 0, 1]),
            (vec![1, 0, 0], vec![1, 0, 1]),
            (vec![0, 1, 0], vec![1, 1, 1]),
            (vec![3, 2, 0, 1, 0], vec![1, 0, 3, 0, 2]),
            (
                vec![5, 0, 4, 0, 3, 0, 2, 0, 1],
                vec![1, 0, 2, 0, 3, 0, 4, 0, 5],
            ),
            (vec![8, 7, 6, 5, 4, 3, 2, 1], vec![1, 2, 3, 4, 5, 6, 7, 8]),
        ];

        for (bottom, top) in pin_list_pairs {
            let mut test_router = Router {
                bottom_pin_list: bottom,
                top_pin_list: top,
                minimum_jog_length: 10,
                steady_net_constant: 5,
                current_column: 0,
                channel_length: 3,
                channel_width: 4,
            };

            let _ = test_router.route();
        }
    }
}
