#! /usr/bin/env python3
from itertools import combinations, product, chain
import networkx as nx


class Router:
    def __init__(self, T, B, initial_channel_width = None, minimum_jog_length = 1, steady_net_constant = 10):
        self.T = T
        self.B = B
        assert len(T) == len(B)
        self.minimum_jog_length = minimum_jog_length
        self.steady_net_constant = steady_net_constant

        self.channel_length = len(T)

        if initial_channel_width is None:
            self.initial_channel_width = self.compute_density()
        else:
            assert initial_channel_width > 0
            self.initial_channel_width = initial_channel_width
        
        self.channel_width = self.initial_channel_width
        self.current_column = 0
        
        nets = set(self.T).union(set(self.B))
        self.Y = dict((i, set()) for i in nets if i != 0)

        # Graph to contain the final routing and intermediate segments
        self.G = nx.Graph()
        # Node ID management
        self.node_id_counter = 0
        self.pos_to_id = {} # Maps (x,y) to integer node_id
        # note: this code used to use the (x,y) tuple as the node identifier,
        # but this gave interoperability issues and would be a pint to port to Rust later on.

    def reset(self, initial_channel_width=None, minimum_jog_length=None, steady_net_constant=None, T=None, B=None):
        # Use stored initial parameters if not overridden
        if initial_channel_width:
            self.initial_channel_width = initial_channel_width
        if minimum_jog_length:
            self.minimum_jog_length = minimum_jog_length
        if steady_net_constant:
            self.steady_net_constant = steady_net_constant
        if T is not None:
            self.T = T
        if B is not None:
            self.B = B

        self.channel_length = len(self.T)
        self.current_column = 0
        nets = set(self.T).union(set(self.B)).difference({0})
        self.Y = dict((i, set()) for i in nets) 
        self.G = nx.Graph()
        self.node_id_counter = 0
        self.pos_to_id = {}

    def create_terminals(self, edges):
        pass

    # Getters
    # -------
    @property
    def all_tracks(self):
        return set(range(1, self.channel_width + 1))
    
    @property
    def occupied_tracks(self):
        return set.union(*self.Y.values())
    
    @property
    def free_tracks(self):
        return self.all_tracks - self.occupied_tracks
    
    @property
    def split_nets(self):
        return sorted(set([net for net, tracks in self.Y.items() if len(tracks) > 1]))
    
    @property
    def vertical_wiring(self):
        v_wires = []
        for u_id, v_id, data in self.G.edges(data=True):
            net = data.get('net')
            if net is None: # Should not happen for routing segments
                continue

            u_pos = self.get_node_pos(u_id)
            v_pos = self.get_node_pos(v_id)
            if not u_pos or not v_pos: # Should not happen if graph is consistent
                continue

            x1, y1 = u_pos
            x2, y2 = v_pos

            if x1 == x2 and x1 == self.current_column:
                v_wires.append((min(y1, y2), max(y1, y2), net))
        return v_wires

    @property
    def pins(self):
        # Get the nets of any unrouted pins in the current column
        # Returns a tuple (top, bottom) where top and bottom may be int or None
        x = self.current_column
        if x >= self.channel_length:
            return (None, None)
        y_t, y_b = self.channel_width + 1, 0

        top_net = self.T[x] if self.T[x] > 0 else None
        bottom_net = self.B[x] if self.B[x] > 0 else None

        top = top_net if not self.has_node_at_pos(x, y_t) else None
        bottom = bottom_net if not self.has_node_at_pos(x, y_b) else None
        
        return (top, bottom)
    
    @property
    def finished(self):
        return (not self.next_pin() 
                and len(self.occupied_tracks) == 0 
                and self.pins == (None, None))

    
    # Generic methods
    # ---------------
    def powerset(self, elements, exclude_empty=False):
        # Full combinatorial search with all combinations from length 0 to len(elements)
        # (includes the empty set)
        for n in range(1 if exclude_empty else 0, len(elements) + 1):
            for combo in combinations(elements, n):
                yield combo

    def overlaps(self, pairs):
        """
        Tests for overlaps between any pairs in a list of pairs.
        """
        if len(pairs) == 1:
            return False
        
        # Each pair is a tuple of (start, stop), sort them by start
        pairs = sorted(pairs, key=lambda x: x[0])

        # Check for overlaps
        for (_, stop1), (start2, _) in zip(pairs, pairs[1:]):
            # An overlap occurs if the next segment starts before the previous one ends
            if stop1 >= start2: #somewhat controversial
                return True
        return False
    
    def contiguous(self, pairs):
        """
        Tests if the given pairs of segments are contiguous.
        """
        if len(pairs) == 1:
            return True
        
        # Each pair is a tuple of (start, stop), sort them by start
        pairs = sorted(pairs, key=lambda x: x[0])

        # Check if the pairs are contiguous
        for (_, stop1), (start2, _) in zip(pairs, pairs[1:]):
            if stop1 != start2:
                return False
        return True
    
    def colinear(self, p1, p2, p3):
        """
        Tests if the points, given as 3 tuples (x, y), are collinear.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1)
    
    # Keeping this here for reference when porting this to Rust
    # def rename_node(self, old_node, new_node):
    #     """
    #     Rename a node in the graph by replacing it with a new node with the same attributes. 
    #     """
    #     attributes = self.G.nodes[old_node]
    #     self.G.add_node(new_node, **attributes)
    #     for (x1, y1), (x2, y2), edge_attr in self.G.edges(old_node, data=True):
    #         if (x1, y1) == old_node:
    #             self.G.add_edge(new_node, (x2, y2), **edge_attr)
    #         else:
    #             self.G.add_edge((x1, y1), new_node, **edge_attr)
    #     # Removing the old node also removes all edges connected to it
    #     self.G.remove_node(old_node)

    def _get_node_orientation(self):
        for node_id in self.G.nodes():
            pos = self.get_node_pos(node_id)
            if not pos: continue # Should not happen
            x, y = pos
            neighbors = self.G.neighbors(node_id)
            north, south, east, west = False, False, False, False
            for neighbor_id in neighbors:
                neighbor_pos = self.get_node_pos(neighbor_id)
                if not neighbor_pos: continue
                nx, ny = neighbor_pos
                # Use Router's coordinate system (y increases upwards)
                if ny > y: north = True
                if ny < y: south = True
                # Use Router's coordinate system (x increases rightwards)
                if nx > x: east = True
                if nx < x: west = True
            # Store the orientation in the node attribute
            self.G.nodes[node_id]['ports'] = (north, south, east, west)

    # TODO: drop this and perform the simplification in route_layer.py instead
    def _simplify(self):
        """
        Simplifies the graph by building a new graph, using the pre-calculated 
        'ports' attribute (node orientation N, S, E, W) on the original graph nodes
        to identify and contract segments.
        """
        # Assumes self.get_node_orientation() has been called beforehand 
        # to populate the 'ports' attribute on self.G nodes.
        if not self.G or self.G.number_of_edges() == 0:
            return

        simplified_G = nx.Graph()
        processed_segments = set()

        # Define straight orientations
        STRAIGHT_VERTICAL = (True, True, False, False)
        STRAIGHT_HORIZONTAL = (False, False, True, True)

        # 1. Identify and add all critical nodes (non-straight) using the 'ports' attribute
        critical_nodes = set()
        original_nodes = list(self.G.nodes()) # Still need a stable list to iterate over
        for node_id in original_nodes:
            # Access the pre-calculated orientation from the node attribute
            orientation = self.G.nodes[node_id].get('ports', (False, False, False, False)) 
            
            if orientation not in [STRAIGHT_HORIZONTAL, STRAIGHT_VERTICAL]:
                critical_nodes.add(node_id)
                # Add critical node and copy its attributes (including 'ports')
                simplified_G.add_node(node_id, **self.G.nodes[node_id]) 
        
        # Handle case where graph might be a single loop of straight segments
        if not critical_nodes and self.G.number_of_nodes() > 0:
            start_node_id = original_nodes[0] 
            critical_nodes.add(start_node_id)
            simplified_G.add_node(start_node_id, **self.G.nodes[start_node_id])

        # 2. Iterate through critical nodes and trace segments using the 'ports' attribute
        for start_node_id in critical_nodes:
            for neighbor_id in list(self.G.neighbors(start_node_id)):
                
                if neighbor_id in critical_nodes:
                    # Direct connection
                    segment_endpoints = tuple(sorted((start_node_id, neighbor_id)))
                    if segment_endpoints not in processed_segments:
                        net_attr = self.G.edges[start_node_id, neighbor_id].get('net')
                        if not simplified_G.has_edge(start_node_id, neighbor_id):
                            simplified_G.add_edge(start_node_id, neighbor_id, net=net_attr)
                        processed_segments.add(segment_endpoints)
                else:
                    # Start of a straight segment
                    prev_id = start_node_id
                    curr_id = neighbor_id
                    segment_net = self.G.edges[prev_id, curr_id].get('net')
                    
                    while True:
                        # Use stored 'ports' attribute
                        curr_orientation = self.G.nodes[curr_id].get('ports', (False, False, False, False))
                        
                        if curr_orientation in [STRAIGHT_HORIZONTAL, STRAIGHT_VERTICAL]:
                            neighbors_of_curr = list(self.G.neighbors(curr_id)) 
                            if len(neighbors_of_curr) != 2: 
                                end_node_id = curr_id
                                break
                            next_node_id = neighbors_of_curr[0] if neighbors_of_curr[1] == prev_id else neighbors_of_curr[1]
                            
                            next_net = self.G.edges[curr_id, next_node_id].get('net')
                            if next_net != segment_net:
                                end_node_id = curr_id 
                                break 

                            prev_id = curr_id
                            curr_id = next_node_id
                            if curr_id in critical_nodes: 
                                end_node_id = curr_id
                                break
                        else:
                            end_node_id = curr_id 
                            break
                    
                    # Add simplified edge
                    if end_node_id in critical_nodes:
                        segment_endpoints = tuple(sorted((start_node_id, end_node_id)))
                        if segment_endpoints not in processed_segments:
                             if not simplified_G.has_edge(start_node_id, end_node_id):
                                simplified_G.add_edge(start_node_id, end_node_id, net=segment_net)
                             processed_segments.add(segment_endpoints)
                    elif end_node_id not in simplified_G:
                        simplified_G.add_node(end_node_id, **self.G.nodes[end_node_id])

        # Replace the old graph with the simplified one
        self.G = simplified_G

    # Helper methods for node ID and coordinate management
    def add_node_at_pos(self, x, y, **attrs):
        """Add a node at specific coordinates, or return existing if present.
        Coordinates are stored as 'pos' attribute.
        Returns the integer node ID.
        """
        if (x,y) in self.pos_to_id:
            node_id = self.pos_to_id[(x,y)]
            return node_id
        else:
            node_id = self.node_id_counter
            self.pos_to_id[(x,y)] = node_id
            
            attrs.update({'pos': (x,y)})
            
            self.G.add_node(node_id, **attrs)
            self.node_id_counter += 1
            return node_id

    def get_node_id(self, x, y):
        """Generate or retrieve a node ID from x,y coordinates"""
        return self.pos_to_id.get((x, y))

    def get_node_pos(self, node_id):
        """Retrieve coordinates for a given node ID"""
        if node_id is None or node_id not in self.G.nodes:
            return None
        return self.G.nodes[node_id].get('pos')

    def has_node_at_pos(self, x, y):
        """Check if a node exists at the given coordinates"""
        return (x,y) in self.pos_to_id

    # Specialized methods
    # --------------------
    def track_to_net(self, track):
        if track in self.occupied_tracks:
            return next(k for k,v in self.Y.items() if track in v)
        elif track in self.free_tracks:
            return None
        else:
            raise ValueError(f"There is no track {track} in the channel")

    def next_pin(self, net=None, side=None):
        if net:
            top = [i for (i, n) in enumerate(self.T) if n == net and i > self.current_column]
            bottom = [i for (i, n) in enumerate(self.B) if n == net and i > self.current_column]
        else:
            top = [i for (i, n) in enumerate(self.T) if i > self.current_column]
            bottom = [i for (i, n) in enumerate(self.B) if i > self.current_column]
        
        if side == 'T' and len(top) > 0:
            return min(top)
        elif side == 'B' and len(bottom) > 0:
            return min(bottom)
        elif side is None and (len(top) > 0 or len(bottom) > 0):
            return min(top + bottom)
        else:
            return None
        
    def classify_net(self, net):
        next_top = self.next_pin(net, 'T') # k in the paper
        next_bottom = self.next_pin(net, 'B')

        if next_top and (not next_bottom or (next_bottom >= next_top + self.steady_net_constant)):
            return 'rising'
        elif next_bottom and (not next_top or (next_top >= next_bottom + self.steady_net_constant)):
            return 'falling'
        else:
            return 'steady'
            
    def compute_density(self):
        max_density = 0
        
        # Check density at each possible column position
        for alpha in range(self.channel_length):
            crossing_nets = set()
            
            # Find nets with pins to the left of position e
            left_nets = set()
            for i in range(alpha):
                if self.T[i] != 0:
                    left_nets.add(self.T[i])
                if self.B[i] != 0:
                    left_nets.add(self.B[i])
            
            # Find nets with pins to the right of position e
            right_nets = set()
            for i in range(alpha, self.channel_length):
                if self.T[i] != 0:
                    right_nets.add(self.T[i])
                if self.B[i] != 0:
                    right_nets.add(self.B[i])
            
            # Crossing nets are those that appear on both sides
            crossing_nets = left_nets.intersection(right_nets)
            density = len(crossing_nets)
            
            if density > max_density:
                max_density = density
        
        return max_density
        
    # Step 1: Make feasible top and bottom connections in minimal manner
    # ------------------------------------------------------------------
    def add_vertical_wire(self, net, from_track, to_track):
        # Ensure that y1 < y2:
        y1 = min(from_track, to_track)
        y2 = max(from_track, to_track)
        node1_id = self.add_node_at_pos(self.current_column, y1)
        node2_id = self.add_node_at_pos(self.current_column, y2)
        self.G.add_edge(node1_id, node2_id, net=net)

    def connect_pins(self):
        top_net = self.T[self.current_column]
        bottom_net = self.B[self.current_column]
        y1 = 0
        y2 = self.channel_width + 1

        # Use the vertical wires to check if the pins have already been connected
        bottom_connected = any(y1 == 0 for y1, y2, net in self.vertical_wiring)
        top_connected = any(y2 == self.channel_width + 1 for y1, y2, net in self.vertical_wiring)

        if top_connected and bottom_connected:
            return

        # Special case: 
        #     if there are no empty tracks, and net Ti = Bi =/=0 is a net which has connections in this column only, 
        #     then run a vertical wire from top to bottom of this column
        if (top_net != 0 and bottom_net != 0
            and top_net == bottom_net
            and len(self.occupied_tracks) == self.channel_width
            and self.Y[top_net] == set()
            and self.next_pin(top_net) is None):

            # Vertical wire from bottom to top
            self.add_vertical_wire(top_net, y1, y2)
            return 
        
        # Find the nearest track for the top and/or bottom pins
        if bottom_net != 0 and not bottom_connected:
            possible_tracks = self.free_tracks.union(self.Y[bottom_net])
            bottom_track = min(possible_tracks) if possible_tracks else None
        if top_net != 0 and not top_connected:
            possible_tracks = self.free_tracks.union(self.Y[top_net])
            top_track = max(possible_tracks) if possible_tracks else None

        # If there is overlap, only keep the shortest vertical wire,
        # the other pin will be connected when the channel is widened.
        if (bottom_net != 0 and not bottom_connected and bottom_track is not None
            and top_net != 0 and not top_connected and top_track is not None):
            # Check if the same net is connecting top and bottom
            if top_net == bottom_net:
                # Same net (T[i] == B[i] != 0): Allow overlap 
                self.Y[bottom_net].add(bottom_track) # bottom_net == top_net
                self.Y[top_net].add(top_track)
                self.add_vertical_wire(bottom_net, 0, bottom_track)
                self.add_vertical_wire(top_net, top_track, self.channel_width + 1)
            else:
                # Different nets:
                if bottom_track < top_track:
                    self.Y[bottom_net].add(bottom_track)
                    self.Y[top_net].add(top_track)
                    self.add_vertical_wire(bottom_net, 0, bottom_track)
                    self.add_vertical_wire(top_net, top_track, self.channel_width + 1)
                else:
                    # Overlap, only keep the shortest vertical wire
                    # Compare vertical distances: bottom pin vs top pin
                    if bottom_track < (self.channel_width + 1 - top_track):
                        # Bottom connection is shorter
                        self.Y[bottom_net].add(bottom_track)
                        self.add_vertical_wire(bottom_net, 0, bottom_track)
                    else:
                        # Top connection is shorter or equal
                        self.Y[top_net].add(top_track)
                        self.add_vertical_wire(top_net, top_track, self.channel_width + 1)

        elif bottom_net != 0 and not bottom_connected and bottom_track is not None:
            self.Y[bottom_net].add(bottom_track)
            self.add_vertical_wire(bottom_net, 0, bottom_track)
        elif top_net != 0 and not top_connected and top_track is not None:
            self.Y[top_net].add(top_track)
            self.add_vertical_wire(top_net, top_track, self.channel_width + 1)

    # Step 2: Free as many tracks as possible by collapsing split nets
    # ----------------------------------------------------------------
    def generate_jog_patterns(self):
        # Generate all possible jog patterns for the current column
        # Returns a pattern as a list of jogs, grouped by the net they belong to:
        # [((track1, track2), (track3, track4), ...), ((track5, track6), (track7, track8), ...), ...]
        # (This also includes empty groups to keep the distinction between nets)
        
        jogs = []
        for net in sorted(self.split_nets):
            # Generate all possible jogs for this net (non-overlapping)
            track_list = sorted(self.Y[net])
            net_jogs = tuple(zip(track_list, track_list[1:]))
            jogs.append(net_jogs)

        # To build the patterns we take the cartesian product of the power sets of each net
        patterns = []
        for pattern in product(*(self.powerset(net_jogs, exclude_empty=True) for net_jogs in jogs)):
            if self.validate_jog_pattern(pattern):
                patterns.append(pattern)
        return patterns

    def validate_jog_pattern(self, pattern):
        # Pattern is a list of jogs, grouped by net:
        # [((track1, track2), (track3, track4), ...), ((track5, track6), (track7, track8), ...), ...]
        # Check if the jogs in the pattern are valid by testing for overlaps
        # between the jogs of different nets. This is a two-step combination:
        # 1) each net is checked against all other nets
        # 2) all jogs from one net are checked against all jogs from the other net
        # Returns True if valid, False otherwise 

        for net1_jogs, net2_jogs in combinations(pattern, 2):
            # Empty groups cause the product to be empty, which is valid
            for j1, j2 in product(net1_jogs, net2_jogs):
                low1, high1 = j1
                low2, high2 = j2
                assert low1 < high1 and low2 < high2
                if not (high1 < low2 or high2 < low1):
                    return False
        return True
    
    def evaluate_jogs(self, pattern):
        # pattern is a list of lists of tuples (track1,track2) grouped by net
        # Returns a score as 3 values:
        # 1. Number of tracks freed
        # 2. Outermost split net distance from edge
        # 3. Sum of jog lengths

        # 1) Number of new empty tracks created by the jogs (higher is better)
        # From the paper: "a pattern [of jogs] will free up one track for every jog it contains, 
        # plus one additional track for every net it finishes"
        all_jogs = list(chain(*pattern))
        n_freed = len(all_jogs)

        # The only nets we can finish are the split nets that are still being routed, but don't have an upcoming pin
        almost_finished_nets = [net for net, tracks in self.Y.items() 
                                if (self.next_pin(net) is None) and (len(tracks) > 1)]
        all_jogs = []
        for group in pattern:
            if len(group) == 0:
                continue

            # Flatten the jogs into a single list for use in step 2
            all_jogs.extend(group)

            # If the jogs for that net are not contiguous, it won't fully close out the net
            # so we don't count it.
            if not self.contiguous(group):
                continue

            # Test if there's a net for which we've just freed all the tracks
            # and that there are no pins coming up anymore.
            for net in almost_finished_nets:
                if self.Y[net].issubset(chain(*group)):
                    n_freed += 1
                    break

        
        # 2) Maximize the distance of the outermost split net from the edge
        # Find all split nets that would not be joined by the jogs
        # Then find the outermost track of each of those nets
        # Then take the minimum distance of those outermost tracks from the edge
        net_distances = []
        all_tracks = set(chain(*all_jogs))
        for net in self.split_nets:
            dangling_tracks = self.Y[net].difference(all_tracks)
            if not dangling_tracks:
                continue
            distance_from_bottom = min(dangling_tracks) - 1
            distance_from_top = self.channel_width - max(dangling_tracks)
            net_distances.append(min(distance_from_bottom, distance_from_top))

        # Save a sorted list so that we can also compare the second net etc. 
        distance_ranking = sorted(net_distances) 

        # 3) Minimize the total length of the jogs
        jog_length_sum = sum(y2 - y1 for (y1, y2) in all_jogs)
        
        return n_freed, distance_ranking, jog_length_sum
    
    def compare_scores(self, score1, score2):
        # Find the best pattern with multiple tiebreakers
        # returns True if score1 is better than score2

        n_freed1, distance_ranking1, jog_length_sum1 = score1
        n_freed2, distance_ranking2, jog_length_sum2 = score2

        # Maximize the number of tracks freed
        if n_freed1 != n_freed2:
            return n_freed1 > n_freed2
        
        # Maximize the distance of the outermost split net from the edge
        # If the distance is the same, then compare the second outermost net etc.
        if distance_ranking1 != distance_ranking2:
            for d1, d2 in zip(distance_ranking1, distance_ranking2):
                if d1 != d2:
                    return d1 > d2
        
        # Maximize the total length of the jogs
        return jog_length_sum1 > jog_length_sum2

    def collapse_split_nets(self):
        # Finds an optimal pattern of jogs between tracks holding split nets
        if len(self.split_nets) == 0:
            return
        
        # Generate all legal jog combinations for the current column
        jog_patterns = self.generate_jog_patterns()
        if len(jog_patterns) == 0:
            return

        # Filter out any jog pattern that would overlap with an existing vertical
        # wire from a DIFFERENT net in this column.
        filtered_patterns = []
        existing_verticals = []  # tuples: (y_low, y_high, net)
        for u_id, v_id, data in self.G.edges(data=True):
            u_pos = self.get_node_pos(u_id)
            v_pos = self.get_node_pos(v_id)
            if not u_pos or not v_pos: continue # Should not happen

            x1, y1 = u_pos
            x2, y2 = v_pos
            if x1 == x2 == self.current_column:
                existing_verticals.append((min(y1, y2), max(y1, y2), data.get('net')))

        for pattern in jog_patterns:
            valid = True
            for idx, group in enumerate(pattern):
                net = self.split_nets[idx]  # group corresponds to this net
                for y1, y2 in group:
                    jog_pair = (min(y1, y2), max(y1, y2))
                    for v_y1, v_y2, v_net in existing_verticals:
                        if v_net != net and self.overlaps([jog_pair, (v_y1, v_y2)]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                filtered_patterns.append(pattern)

        jog_patterns = filtered_patterns
        if len(jog_patterns) == 0:
            return

        # Test all combinations of jogs to find the pattern that creates the most empty tracks
        best_pattern = None
        # The score is a tuple of 3 values (#tracks freed and tiebreakers)
        best_score = [0, [], self.channel_width] # This will always lose
        for combo in jog_patterns:
            score = self.evaluate_jogs(combo)
            if self.compare_scores(score, best_score):
                best_score = score
                best_pattern = combo                

        if best_pattern is None:
            print("No valid patterns found: ", jog_patterns)
            return

        # The groups are still in the same order as the split nets (which stays sorted)
        for (net, group) in zip(self.split_nets, best_pattern):
            # Add a vertical segment to the net and free up one of the tracks
            for y1, y2 in group:
                self.add_vertical_wire(net, y1, y2)
                self.Y[net].remove(y1)
            # If the net is closed, y2 will be removed in a later step
            
    # Step 3: Add jogs to reduce the range of split nets
    # --------------------------------------------------
    def scout(self, net, track, goal):
        # Find the closest reachable track in the direction of the goal (another track on the same net, or an empty track).
        # Assumes that there are not other tracks of the same net in the way
        # Returns the position of that track number if successful, or the original track if not.
        if goal > track:
            tracks = range(track+1, goal+1)
        elif goal < track:
            tracks = range(track-1, goal-1, -1)
        else:
            return track

        # We scan the tracks in the direction of a conductor on the same net,
        # and keep a marker of the last reachable track.
        marker = track
        for i in tracks:
            # If the vertical layer is occupied, we have to stop the search.
            if any(min(y1, y2) <= i <= max(y1, y2) for y1, y2, _ in self.vertical_wiring):
                break

            # If the horizontal layer is occupied we can jump over it but not land there 
            if i in self.occupied_tracks and i not in self.Y[net]:
                continue

            # If we made it this far, we can record the index of this iteration in the marker variable
            marker = i

        return marker
    
    def jog(self, net, track, goal):
        # Jog the net from track to as close as possible to goal
        destination = self.scout(net, track, goal)

        if destination != track:
            self.Y[net].remove(track)
            self.Y[net].add(destination)
            self.add_vertical_wire(net, track, destination)
        return destination

    def compress_split_net(self, net):
        # For split nets that weren't collapsed, try to move the tracks closer to each other:
        #  - jog the lowest track up as far as possible 
        #  - jog the highest track down as far as possible
        # To find the correct open spot we process the column cell by cell, on both layers

        tracks = sorted(list(self.Y[net]))
        
        # 1) Attempt to jog the lowest track up as far as possible 
        low_track, goal = tracks[0], tracks[1]
        low_marker = self.scout(net, low_track, goal)

        # 2) Attempt to jog the highest track down as far as possible
        high_track, goal = tracks[-1], tracks[-2]
        high_marker = self.scout(net, high_track, goal)

        # 3) Actually move the tracks if the jog is long enough. Do this after the two attempts 
        # above so that we don't invalidate the markers by moving the tracks.
        # High track
        if abs(high_marker - high_track) >= self.minimum_jog_length:
            self.jog(net, high_track, high_marker)

        # Low track
        if abs(low_marker - low_track) >= self.minimum_jog_length:
            self.jog(net, low_track, low_marker)

    # Step 4: Add jogs to raise rising nets and lower falling nets
    # ------------------------------------------------------------
    def push_unsplit_nets(self):
        x = self.current_column
        # We look specifically for nets that are not split and have a pin coming up
        upcoming_pins = set(self.T[x:] + self.B[x:])
        nets_to_jog = [net for net in upcoming_pins 
                       if net != 0 
                       and len(self.Y[net]) == 1 ]
                
        # Determine where to push the nets to
        track_distances  = [] 
        for net in nets_to_jog:
            try:
                track = next(iter(self.Y[net]))
            except StopIteration:
                print("No tracks found for net: ", net)
                continue
            if self.classify_net(net) == 'rising':
                goal = self.channel_width
            elif self.classify_net(net) == 'falling':
                goal = 1
            else:
                continue
            # Record the achievable distance to the goal track
            destination = self.scout(net, track, goal)    
            distance = abs(track - destination)
            if distance >= self.minimum_jog_length:
                track_distances.append((distance, net, track, goal))

        # Execute longer jogs first
        track_distances.sort(key=lambda x: x[0], reverse=True)
        for _, net, track, goal in track_distances:
            destination = self.jog(net, track, goal)


    # Step 5: Widen channel if needed to make previously not feasible top or bottom connections
    # -----------------------------------------------------------------------------------------
    def widen_channel(self, from_side=None):
        # Inserts a new track which must be: 
        # (a) reachable from the top or bottom
        # (b) as close as possible to the middle of the channel
        # If the track x is selected, then the old tracks x, x+1, ... will be moved up to x+1, x+2, ...
        # Note: we are assuming that this function is only called when there is no space left on the channel

        mid_track = round(self.channel_width / 2) + 1 # +1 because tracks are indexed from 1 to channel_width

        # Find a position for the new track that is as close to the middle as possible,
        # and that is accessible from the pins without violating a vertical constraint.
        current_vertical_wires = self.vertical_wiring # Call property once
        if not current_vertical_wires:
            min_start = 1
            max_end = self.channel_width # Before widening, tracks go up to channel_width
        else:
            min_start = min(current_vertical_wires, key=lambda item: item[0])[0] # y1
            max_end = max(current_vertical_wires, key=lambda item: item[1])[1]   # y2
        
        new_track_candidate_bottom = min(min_start, mid_track)
        new_track_candidate_top = max(max_end + 1, mid_track)


        if from_side == 'B':
            # Moving upwards from the bottom: the start of the first vertical wire, 
            # or the middle, whichever comes first. New track is inserted AT this position.
            new_track = new_track_candidate_bottom
        elif from_side == 'T':
            # Moving downwards from the top: the end of the last vertical wire + 1,
            # or the middle, whichever comes first. New track is inserted AT this position.
            new_track = new_track_candidate_top
        elif from_side is None: # Changed from 'from_side == None' for style
            new_track = mid_track
        else:
            raise ValueError("Invalid side (only 'T', 'B', or None are allowed)")
        
        # Ensure new_track is within reasonable bounds (1 to channel_width + 1)
        # If inserting at new_track, all tracks >= new_track shift up.
        # new_track is 1-indexed.
        new_track = max(1, min(new_track, self.channel_width + 1))

        self.channel_width = self.channel_width + 1

        # Update the active assignments for all tracks above the new track,
        # starting from the top down so we don't overwrite any existing assignments
        Y_new = {net: set() for net in self.Y.keys()}
        for net, tracks in self.Y.items():
            for track in tracks:
                if track >= new_track:
                    Y_new[net].add(track + 1)
                else:
                    Y_new[net].add(track)
        self.Y = Y_new
                
        # Update the graph by moving up any nodes that are now at or above the new track's position.
        # Also update the pos_to_id mapping.
        updated_pos_to_id = {}
        nodes_to_update = list(self.G.nodes()) # Iterate over a copy of node IDs

        for node_id in nodes_to_update:
            pos = self.get_node_pos(node_id)
            if not pos: continue # Should not happen for existing nodes

            x, y = pos
            if y >= new_track:
                new_y = y + 1
                self.G.nodes[node_id]['pos'] = (x, new_y)
                # Old coord mapping will be removed when rebuilding updated_pos_to_id
            # else: node coordinates remain unchanged

        # Rebuild pos_to_id from the updated graph node attributes
        self.pos_to_id.clear()
        for node_id, data in self.G.nodes(data=True):
            if 'pos' in data:
                 self.pos_to_id[data['pos']] = node_id
            # else: # This case should ideally not happen if all nodes have pos
            #    print(f"Warning: Node {node_id} has no 'pos' attribute during widen_channel.")

    def extend_nets(self):
        # Only extend nets that either are split or have a pin coming up
        for net, tracks in self.Y.items():
            if len(tracks) == 1 and self.next_pin(net) is None:
                # Clear the Y dict entry for this net
                self.Y[net] = set()
            else:
                # Update the graph for each track
                for track in tracks:
                    node1_id = self.add_node_at_pos(self.current_column, track)
                    node2_id = self.add_node_at_pos(self.current_column + 1, track)
                    self.G.add_edge(node1_id, node2_id, net=net)
                                    
        # Update the channel length if needed
        self.channel_length = max(self.channel_length, self.current_column+1)

    def route(self):
        # Route the nets, returns the final graph

        # The algorithm will dynamically extend the channel as needed,
        # but we don't want it to extend indefinitely. This is in case
        # the channel is blocked by a net that cannot be routed.
        max_length = self.channel_length * 1.5

        while not self.finished:
            x = self.current_column
            # 1) Connect the pins
            if x < self.channel_length:
                self.connect_pins()
            # 2) Collapse split nets to free up tracks
            self.collapse_split_nets()

            # 3) Compress remaining split nets to narrow their range
            for net in self.split_nets:
                self.compress_split_net(net)

            # 4) Add jogs to raise rising nets and lower falling nets
            # We look specifically for nets that are not split and have a pin coming up
            self.push_unsplit_nets()

            # 5) Widen the channel if we were not able to route pins earlier because of space constraints
            top_net, bottom_net= self.pins
            if top_net is not None:
                self.widen_channel(from_side='T')
                self.connect_pins()

            if bottom_net is not None:
                self.widen_channel(from_side='B')
                self.connect_pins()

            # 6) Extend nets to the next column and advance the column pointer
            self.extend_nets()
            self.current_column += 1

            # Failsafe: if we keep extending the channel without making progress,
            # we'll stop anyway
            if self.channel_length >= max_length:
                break

        # Store how each node is oriented with respect to its neighbors
        self._get_node_orientation()

        # Simplify the graph to remove any redundant edges
        self._simplify()
        
        return self.G

    def route_and_retry(self, tries_left=10):
        # If routing fails, try again with a wider channel:
        # (the other parameters tend to be the problem and make the output worse when deviating from the defaults)
        if tries_left == 0:
            print(f"Failed to route: T = {self.T}, B = {self.B}")
            print(f"  initial_channel_width = {self.initial_channel_width}")
            print(f"  minimum_jog_length = {self.minimum_jog_length}")
            raise ValueError("Failed to route the edges")
        try:
            return self.route()
        except ValueError as e:
            print(e)
            print(f"Retrying with a wider channel... {tries_left} tries left")
            # Increase the initial channel width and try again
            self.reset(initial_channel_width=self.initial_channel_width + 1)
            return self.route_and_retry(tries_left-1)
        

    

    

class Plotter:
    def __init__(self, G, hscale=2, vscale=1):
        # Transform the graph from the router's coordinate system to the plotter's
        # - transposed (left-to-right instead of top-to-bottom)
        # - scaled (hscale, vscale)
        self.plot_G = nx.Graph() 
        self.hscale = hscale
        self.vscale = vscale

        node_id_to_plot_pos = {}

        for node_id, data in G.nodes(data=True):
            if 'pos' not in data:
                # print(f"Warning: Router graph node {node_id} missing 'pos'. Skipping in Plotter.")
                continue
            x_router, y_router = data['pos']
            
            # Transpose and scale for plotter's coordinate system
            # Router: (x increases right, y increases up)
            # Plotter Display: (x increases right, y increases up on grid, but rendering reverses rows)
            # Plotter node pos: (scaled_y_router, scaled_x_router) to match original intent
            plot_node_pos = (y_router * self.hscale, x_router * self.vscale)
            node_id_to_plot_pos[node_id] = plot_node_pos
            
            # Copy all attributes from router node to plotter node
            self.plot_G.add_node(plot_node_pos, **data) 

        for u_id, v_id, edge_data in G.edges(data=True):
            if u_id in node_id_to_plot_pos and v_id in node_id_to_plot_pos:
                plot_u_pos = node_id_to_plot_pos[u_id]
                plot_v_pos = node_id_to_plot_pos[v_id]
                self.plot_G.add_edge(plot_u_pos, plot_v_pos, **edge_data)
            # else:
                # print(f"Warning: Edge ({u_id}, {v_id}) skipped in Plotter due to missing node(s).")

        self.grid = self.initialize_grid()

    def initialize_grid(self):
        # Grid dimensions: graph bounding box
        if not self.plot_G.nodes(): # Handle empty graph
            self.grid = [[]]
            return self.grid
            
        all_plot_x_pos = [node[0] for node in self.plot_G.nodes()]
        all_plot_y_pos = [node[1] for node in self.plot_G.nodes()]

        min_x = min(all_plot_x_pos + [0]) if all_plot_x_pos else 0
        max_x = max(all_plot_x_pos) if all_plot_x_pos else 0
        min_y = min(all_plot_y_pos + [0]) if all_plot_y_pos else 0
        max_y = max(all_plot_y_pos) if all_plot_y_pos else 0

        # Initialize with spaces
        grid_width = (max_x - min_x + 1)
        grid_height = (max_y - min_y + 1)

        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
        return grid

    def render_text_graph(self):
        grid_height = len(self.grid)
        grid_width = len(self.grid[0]) if grid_height > 0 else 0
        if grid_height == 0 or grid_width == 0: # Handle empty grid
            return ""

        # Determine graph bounds for the grid
        all_plot_x_pos = [node[0] for node in self.plot_G.nodes()]
        all_plot_y_pos = [node[1] for node in self.plot_G.nodes()]
        
        min_x = min(all_plot_x_pos, default=0) if all_plot_x_pos else 0
        min_y = min(all_plot_y_pos, default=0) if all_plot_y_pos else 0

        # 1. Draw horizontal edges first
        for u, v in self.plot_G.edges():
            # Graph coordinates (these are plot_pos from self.plot_G)
            x1, y1 = u
            x2, y2 = v

            # Map to grid coordinates (adjusting for min_x, min_y)
            grid_x1, grid_y1 = x1 - min_x, y1 - min_y
            grid_x2, grid_y2 = x2 - min_x, y2 - min_y

            # Ensure coordinates are within grid bounds
            if not (0 <= grid_y1 < grid_height and 0 <= grid_x1 < grid_width and \
                    0 <= grid_y2 < grid_height and 0 <= grid_x2 < grid_width):
                continue

            if y1 == y2: # Horizontal segment
                char = '─'
                # Iterate over grid columns
                for grid_x in range(min(grid_x1, grid_x2) + 1, max(grid_x1, grid_x2)):
                     if 0 <= grid_x < grid_width:
                        self.grid[grid_y1][grid_x] = char # Use grid_y1

        # 2. Draw vertical edges (can overwrite horizontal)
        for u, v in self.plot_G.edges():
            # Graph coordinates
            x1, y1 = u
            x2, y2 = v

            # Map to grid coordinates
            grid_x1, grid_y1 = x1 - min_x, y1 - min_y
            grid_x2, grid_y2 = x2 - min_x, y2 - min_y

            if x1 == x2: # Vertical segment
                # Iterate over grid rows
                for grid_y in range(min(grid_y1, grid_y2) + 1, max(grid_y1, grid_y2)):
                    if 0 <= grid_y < grid_height:
                         self.grid[grid_y][grid_x1] = '│' # Use grid_x1

        # 3. Draw nodes (intersections/corners/ends) - this overwrites ends of edges
        for node in self.plot_G.nodes(): # node is (plot_x, plot_y)
            x, y = node
            grid_x, grid_y = x - min_x, y - min_y

             # Ensure node coordinates are within grid bounds
            if not (0 <= grid_y < grid_height and 0 <= grid_x < grid_width):
                print(f"Warning: Node {node} (mapped to grid {grid_x, grid_y}) out of bounds") # Node is plot_pos
                continue

            # Retrieve pre-calculated orientation from node attribute
            original_ports = self.plot_G.nodes[node].get('ports', (False, False, False, False)) 

            # Transpose the ports to match the plotter's coordinate system
            # Original Router ports (N, S, E, W) based on router's (x,y)
            # Plotter node pos are (scaled_router_y, scaled_router_x)
            # So, router's N/S (y-axis) corresponds to plotter's E/W (plot_G node_x component)
            # And router's E/W (x-axis) corresponds to plotter's N/S (plot_G node_y component)
            # Plotter display orientation (char selection) uses (N,S,E,W) relative to its grid display.
            # The mapping given in prompt was: Original (N,S,E,W) -> Plotter (E,W,N,S)
            # Let's stick to the prompt's transposition rule for BOX_CHARS lookup.
            N_router, S_router, E_router, W_router = original_ports
            plotter_orientation_for_char = (E_router, W_router, N_router, S_router) 

            char = self.BOX_CHARS.get(plotter_orientation_for_char, '?')
            self.grid[grid_y][grid_x] = char

        # 4. Prepare Net Labels for Boundaries
        left_labels = {}
        right_labels = {}
        # Determine max_x for the plot_G to check boundary conditions
        plot_G_max_x = max(all_plot_x_pos) if all_plot_x_pos else 0
        
        for u_plot_pos, v_plot_pos, data in self.plot_G.edges(data=True):
            net = str(data.get('net', '?')) # Get net ID as string
            ux_plot, uy_plot = u_plot_pos
            vx_plot, vy_plot = v_plot_pos

            # Map graph y to grid y coordinate
            grid_uy = uy_plot - min_y
            grid_vy = vy_plot - min_y

            # Check against graph boundaries (min_x, max_x) of the plot_G
            if ux_plot == min_x and 0 <= grid_uy < grid_height: left_labels[grid_uy] = net
            if vx_plot == min_x and 0 <= grid_vy < grid_height: left_labels[grid_vy] = net
            if ux_plot == plot_G_max_x and 0 <= grid_uy < grid_height: right_labels[grid_uy] = net
            if vx_plot == plot_G_max_x and 0 <= grid_vy < grid_height: right_labels[grid_vy] = net

        max_left_len = max(len(s) for s in left_labels.values()) if left_labels else 0
        max_right_len = max(len(s) for s in right_labels.values()) if right_labels else 0

        # 5. Format grid to string with labels (reverse rows for intuitive printing)
        output_lines = []
        # Iterate through grid rows from top (index 0) to bottom
        for i, row in enumerate(self.grid):
            # grid row i corresponds to graph y = i + min_y
            current_grid_y = i
            left_label = left_labels.get(current_grid_y, "")
            right_label = right_labels.get(current_grid_y, "")
            # Pad labels
            padded_left = f"{left_label:<{max_left_len}}"
            padded_right = f"{right_label:<{max_right_len}}" # Pad right label as well

            line = "".join(row)
            # Add labels only if they exist, otherwise add padding of spaces
            left_part = f"{padded_left} " if max_left_len > 0 else ""
            right_part = f" {padded_right}" if max_right_len > 0 else ""

            output_lines.append(f"{left_part}{line}{right_part}")

        # Return lines, joining top-to-bottom which means reversing the list built from iterating grid 0..height-1
        return "\n".join(reversed(output_lines))

    BOX_CHARS = {
        # N S E W
        (False, False, False, False): '?', # Should not happen in connected graph
        # Straights
        (True,  True,  False, False): 'x',#'│', # Should not be left over after simplification
        (False, False, True,  True ): 'x',#'─', # Should not be left over after simplification
        # Corners
        (True,  False, True,  False): '╰',
        (True,  False, False, True ): '╯',
        (False, True,  True,  False): '╭',
        (False, True,  False, True ): '╮',
       # T-junctions
        (True,  True,  True,  False): '├',
        (True,  True,  False, True ): '┤',
        (True,  False, True,  True ): '┴',
        (False, True,  True,  True ): '┬',
        # Cross
        (True,  True,  True,  True ): '┼',
        # Ends (should match straights if node degree is 1)
        (True,  False, False, False): '│', # End pointing North
        (False, True,  False, False): '│', # End pointing South
        (False, False, True,  False): '─', # End pointing East
        (False, False, False, True ): '─', # End pointing West
    }

     
 





if __name__ == '__main__':
    import random
    def random_pins(N, M):
        pins = list(range(N+1))
        pins.extend(random.choices(pins, k = M - N))
        random.shuffle(pins)
        # Add spacing between pins
        pins_spaced = []
        for pin in pins:
            pins_spaced.extend([pin, 0])
        pins_spaced.pop()
        return pins_spaced

    # Set random seed for reproducibility
    random.seed(42)

    with open('edge_router_snapshot.txt', 'w') as f:
        for i in range(10):
            L = random_pins(10, 15)
            R = random_pins(10, 15)
            router = Router(L, R)
            G = router.route()
            output = Plotter(G).render_text_graph()
            f.write(output)
            f.write('\n')

            print(output)
            print('='*60)




    
            


        