#!/usr/bin/env python3

import unittest
import random
from rectilinear_router import Router, Plotter

class TestRouterSkeletons(unittest.TestCase):
    def setUp(self):
        self.T = [1, 2, 3]
        self.B = [3, 2, 1]
        self.router = Router(self.T, self.B)

    def test_init(self):
        # Should initialize with correct T, B, and channel_width
        self.assertEqual(self.router.T, self.T)
        self.assertEqual(self.router.B, self.B)
        self.assertIsInstance(self.router.channel_width, int)
        self.assertGreater(self.router.channel_width, 0)

    def test_reset(self):
        # Should reset current_column and Y
        self.router.current_column = 5
        self.router.Y[1].add(99)
        self.router.reset()
        self.assertEqual(self.router.current_column, 0)
        for tracks in self.router.Y.values():
            self.assertEqual(tracks, set())

    def test_all_tracks_property(self):
        tracks = self.router.all_tracks
        self.assertIsInstance(tracks, set)
        self.assertEqual(tracks, set(range(1, self.router.channel_width + 1)))

    def test_occupied_tracks_property(self):
        # Initially empty
        self.assertIsInstance(self.router.occupied_tracks, set)
        self.assertEqual(self.router.occupied_tracks, set())
        # Occupy a track
        self.router.Y[1].add(2)
        self.assertIn(2, self.router.occupied_tracks)

    def test_free_tracks_property(self):
        # All tracks free at start
        self.assertEqual(self.router.free_tracks, self.router.all_tracks)
        # Occupy a track
        self.router.Y[1].add(2)
        self.assertNotIn(2, self.router.free_tracks)

    def test_split_nets_property(self):
        # No split nets at start
        self.assertEqual(self.router.split_nets, [])
        # Add split net
        self.router.Y[1].update([2, 3])
        self.assertIn(1, self.router.split_nets)

    def test_vertical_wiring_property(self):
        # Should be empty at start
        self.router.reset()
        self.assertEqual(self.router.vertical_wiring, [])
        # Add a vertical edge (manually)
        self.router.G.add_edge((0, 1), (0, 2), net=1)
        self.assertEqual(self.router.vertical_wiring, [(1, 2, 1)]) # net, track, goal
        # Add a vertical wire
        self.router.reset()
        self.router.add_vertical_wire(1, 1, 2)
        self.assertEqual(self.router.vertical_wiring, [(1, 2, 1)])
        self.router.reset()
        self.assertEqual(self.router.vertical_wiring, [])

    def test_finished_property(self):
        # Should not be finished at start
        self.assertFalse(self.router.finished)
        # Simulate finished state
        self.router.current_column = len(self.T)
        for k in self.router.Y:
            self.router.Y[k] = set()
        self.assertTrue(self.router.finished)

    def test_pins(self):
        # Initial state
        self.assertEqual(self.router.pins, (1,3))      

    def test_powerset(self):
        # Should return all subsets including empty set
        elements = [1, 2, 3]
        result = list(self.router.powerset(elements))
        self.assertIn((), result)
        self.assertIn((1, 2, 3), result)
        self.assertEqual(len(result), 2 ** len(elements))

    def test_overlaps(self):
        # No overlap
        pairs = [(1, 2), (3, 4)]
        self.assertFalse(self.router.overlaps(pairs))
        # Overlap
        pairs = [(1, 3), (2, 4)]
        self.assertTrue(self.router.overlaps(pairs))
        # Single pair
        self.assertFalse(self.router.overlaps([(1, 2)]))
        # Borderline overlap
        pairs = [(1, 2), (2, 3)]
        self.assertTrue(self.router.overlaps(pairs))

    def test_contiguous(self):
        # Contiguous
        pairs = [(1, 2), (2, 3), (3, 4)]
        self.assertTrue(self.router.contiguous(pairs))
        # Not contiguous
        pairs = [(1, 2), (3, 4)]
        self.assertFalse(self.router.contiguous(pairs))
        # Single pair
        self.assertTrue(self.router.contiguous([(1, 5)]))

    def test_colinear(self):
        # diagonal:
        # Collinear points
        self.assertTrue(self.router.colinear((0, 0), (1, 1), (2, 2)))
        # Not collinear
        self.assertFalse(self.router.colinear((0, 0), (1, 1), (2, 3)))

    def test_collapse_run(self):        
        pass

    def test_simplify(self):
        pass

    def test_next_pin(self):
        # Setup: T and B with multiple pins for net 1
        self.router.reset(T=[1, 0, 1, 0], B=[0, 1, 0, 1])
        self.router.current_column = 0
        # Next pin for net 1 (any side)
        #self.assertEqual(self.router.next_pin(1), 1)
        # Next pin for net 1, top side
        #self.assertEqual(self.router.next_pin(1, 'T'), 2)
        # Next pin for net 1, bottom side
        self.assertEqual(self.router.next_pin(1, 'B'), 1)
        # Advance column and check again
        self.router.current_column = 2
        self.assertEqual(self.router.next_pin(1), 3)
        self.router.current_column = 3
        self.assertIsNone(self.router.next_pin(1))

    def test_classify_net(self):
        # Use a small steady_net_constant for clarity
        self.router.reset(T=[1, 0, 0, 1], B=[0, 0, 1, 0], steady_net_constant=1)
        self.router.current_column = 0
        # Net 1: next top at 3, next bottom at 2
        # Should be 'falling' (bottom pin is closer)
        self.assertEqual(self.router.classify_net(1), 'falling')
        # Move column to 1, now next top at 3, next bottom at 2
        #self.router.current_column = 1
        self.assertEqual(self.router.classify_net(1), 'falling')
        # Move column to 2, now next top at 3, no next bottom
        self.router.current_column = 2
        self.assertEqual(self.router.classify_net(1), 'rising')
        # Steady case: both next pins are far apart
        self.router.reset(T=[1, 0, 0, 1], B=[0, 0, 0, 1], steady_net_constant=10)
        self.router.current_column = 0
        self.assertEqual(self.router.classify_net(1), 'steady')

    def test_compute_density(self):
        # Simple case: no overlap
        self.router.reset(T=[1, 0, 2], B=[0, 2, 0])
        self.assertEqual(self.router.compute_density(), 1)
        # Overlapping nets
        self.router.reset(T=[1, 2, 1], B=[2, 1, 2])
        self.assertEqual(self.router.compute_density(), 2)
        # All zeros
        self.router.reset(T=[0, 0, 0], B=[0, 0, 0])
        self.assertEqual(self.router.compute_density(), 0)

    def test_connect_pins(self):
        # Case 1: Only top pin present, should assign to a free track and add vertical wire
        self.router.reset(T=[1], B=[0])
        self.router.channel_width = 3
        self.router.current_column = 0
        self.router.Y = {1: set()}
        self.router.connect_pins()
        # Net 1 should occupy a track
        self.assertEqual(len(self.router.Y[1]), 1)
        # There should be a vertical edge in the graph for net 1
        found = any(
            (edge[0][0] == 0 and edge[1][0] == 0 and (edge[0][1] == self.router.channel_width + 1 or edge[1][1] == self.router.channel_width + 1))
            for edge in self.router.G.edges
        )

        self.assertTrue(found)

        # Case 2: Only bottom pin present, should assign to a free track and add vertical wire
        self.router.reset(T=[0], B=[2])
        self.router.channel_width = 3
        self.router.current_column = 0
        self.router.Y = {2: set()}
        self.router.connect_pins()
        self.assertEqual(len(self.router.Y[2]), 1)
        found = any(
            (edge[0][0] == 0 and edge[1][0] == 0 and (edge[0][1] == 0 or edge[1][1] == 0))
            for edge in self.router.G.edges
        )
        self.assertTrue(found)

        # Case 3: Both pins present, both nets different, should assign both
        self.router.reset(T=[1], B=[2])
        self.router.channel_width = 3
        self.router.current_column = 0
        self.router.Y = {1: set(), 2: set()}
        self.router.connect_pins()
        self.assertEqual(len(self.router.Y[1]), 1)
        self.assertEqual(len(self.router.Y[2]), 1)

        # Case 4: Both pins present, same net, all tracks occupied, should add vertical wire from top to bottom
        self.router.reset(T=[3], B=[3])
        self.router.channel_width = 0
        self.router.current_column = 0
        self.router.Y = {3: set()}
    
        self.router.connect_pins()
        # Should have a vertical wire from 0 to channel_width+1
        found = any(
            (edge[0][1] == 0 and edge[1][1] == self.router.channel_width + 1) or
            (edge[1][1] == 0 and edge[0][1] == self.router.channel_width + 1)
            for edge in self.router.G.edges
        )
        self.assertTrue(found)

    def test_generate_jog_patterns(self):
        # Base config
        patterns = self.router.generate_jog_patterns()
        self.assertEqual(patterns, [()])

        # Simple case
        self.router.reset()
        self.router.T = [0,0,1]
        self.router.B = [0,0,0]
        self.router.channel_width = 6
        self.router.Y[1].add(2)
        self.router.Y[1].add(3)
        self.router.Y[1].add(5)
        self.assertEqual(self.router.split_nets, [1])
        patterns = self.router.generate_jog_patterns()
        # patterns is a list of individual patterns,
        # each of which is a list corresponding to a net,
        # each of which is a list of their jogs
        # each of which is a tuple of two tracks
        for pattern in patterns:
            self.assertEqual(len(pattern), len(self.router.split_nets))
            for net in pattern:
                for jog in net:
                    self.assertEqual(len(jog), 2)

        self.assertIn((((2, 3),),), patterns)
        self.assertNotIn((((2, 5),),), patterns)
        self.assertIn((((2, 3), (3, 5)),), patterns)

        # Multiple nets
        self.router.reset(T=[0,0,1], B=[0,0,2])
        self.router.channel_width = 6
        self.router.Y[1].add(2)
        self.router.Y[1].add(3)
        self.router.Y[2].add(4)
        self.router.Y[1].add(5)
        self.assertEqual(self.router.split_nets, [1])
        patterns = self.router.generate_jog_patterns()
        for pattern in patterns:
            self.assertEqual(len(pattern), len(self.router.split_nets))
            for net in pattern:
                for jog in net:
                    self.assertEqual(len(jog), 2)
        # 3 patterns because not crossing nets
        self.assertEqual(len(patterns), 3)

        # Multiple nets, with crossing between unsplit nets
        self.router.reset(T=[0,0,1,3], B=[0,0,2,3])
        self.router.channel_width = 6
        self.router.Y[1].add(1)
        self.router.Y[2].add(2)
        self.router.Y[1].add(3)
        self.router.Y[2].add(4)
        self.router.Y[1].add(5)
        self.assertEqual(self.router.split_nets, [1,2])
        patterns = self.router.generate_jog_patterns()
        # 0 patterns because every jog crosses the other net
        self.assertEqual(len(patterns), 0)

        # Multiple nets, with crossing between split nets
        self.router.reset(T=[0,0,1], B=[0,0,2])
        self.router.channel_width = 6
        self.router.Y[1].add(1)
        self.router.Y[2].add(2)
        self.router.Y[1].add(3)
        self.router.Y[2].add(4)
        self.router.Y[1].add(5)
        self.assertEqual(self.router.split_nets, [1,2])
        patterns = self.router.generate_jog_patterns()
        # 0 patterns because every jog crosses the other net
        self.assertEqual(len(patterns), 0)


    def test_validate_jog_pattern(self):
        # Test 1: Single net, single jog (valid)
        self.router.reset(T=[0,0,1], B=[0,0,0])
        self.router.Y = {1: set([2, 4])}
        pattern = (((2, 4),),)
        self.assertTrue(self.router.validate_jog_pattern(pattern))

        # Test 2: Two nets, non-overlapping jogs (valid)
        self.router.reset(T=[0,0,1], B=[0,0,2])
        self.router.Y = {1: set([2, 4]), 2: set([5, 6])}
        pattern = (((2, 4),), ((5, 6),))
        self.assertTrue(self.router.validate_jog_pattern(pattern))

        # Test 3: Two split nets, overlapping jogs (invalid)
        self.router.reset(T=[0,0,1], B=[0,0,2])
        self.router.Y = {1: set([2, 4]), 2: set([3, 5])}
        pattern = (((2, 4),), ((3, 5),)) 
        self.assertFalse(self.router.validate_jog_pattern(pattern))

    def test_evaluate_jogs(self):
        # Test 1: one jog from 2 to 4
        self.router.reset(T=[0,0,1], B=[0,0,0])
        self.router.channel_width = 6
        # Setup: split net 1 on tracks 2, 4
        self.router.Y = {1: set([2, 4])}
        # pattern: one jog from 2 to 4
        pattern = (((2, 4),),)
        score = self.router.evaluate_jogs(pattern)
        # Should free 1 track, no unfinished nets to measure distance for, jog length 2
        self.assertEqual(score[0], 1)
        self.assertEqual(score[1], [])
        self.assertEqual(score[2], 2)

        # Test 2: two jogs from 2 to 4 and 3 to 5, does not close net
        self.router.reset(T=[0,0,1], B=[0,0,2])
        self.router.Y = {1: set([2, 4]), 2: set([3, 5])}
        pattern = (((2, 4),), ((3, 5),))
        score = self.router.evaluate_jogs(pattern)
        # Should free 2 tracks, no unfinished nets to measure distance for, jog lengths 2 and 2
        self.assertEqual(score[0], 2)
        self.assertEqual(score[1], [])
        self.assertEqual(score[2], 4)

        # Test 3: two jogs from 2 to 4 and 3 to 5, closes net
        self.router.reset(T=[0,0,1], B=[0,0,0])
        self.router.Y = {1: set([2, 4]), 2: set([3, 5])}
        pattern = (((2, 4),), ((3, 5),))
        score = self.router.evaluate_jogs(pattern)
        # Should free 2 tracks, no unfinished nets to measure distance for, jog lengths 2 and 2
        self.assertEqual(score[0], 3)
        self.assertEqual(score[1], [])
        self.assertEqual(score[2], 4)

        # Test 4: two jogs from 2 to 4 and 3 to 5, but 4 and 5 cross
        self.router.reset(T=[0,0,1], B=[0,0,2])
        self.router.channel_width = 6
        self.router.Y[1].add(2)
        self.router.Y[2].add(3)
        self.router.Y[1].add(4)
        self.router.Y[2].add(5)

        pattern = (((2, 4),), ((3, 5),))
        score = self.router.evaluate_jogs(pattern)
        # Should free 2 tracks, no unfinished nets to measure distance for, jog lengths 2 and 2
        self.assertEqual(score[0], 2)
        self.assertEqual(score[1], [])
        self.assertEqual(score[2], 4)

       
    def test_compare_scores(self):
        # n_freed, distance_ranking, jog_length_sum
        score1 = (2, [3, 2], 5)
        score2 = (1, [10, 9], 20)
        # score1 should be better (more tracks freed)
        self.assertTrue(self.router.compare_scores(score1, score2))
        # If n_freed is equal, compare distance_ranking
        score3 = (2, [4, 2], 5)
        self.assertFalse(self.router.compare_scores(score1, score3))
        self.assertTrue(self.router.compare_scores(score3, score1))
        # If both n_freed and distance_ranking are equal, compare jog_length_sum
        score4 = (2, [3, 2], 10)
        self.assertTrue(self.router.compare_scores(score4, score1))
        self.assertFalse(self.router.compare_scores(score1, score4))

    def test_collapse_split_nets(self):
        # Test 1: one jog from 2 to 4, does not close net
        self.router.reset(T=[0,0,1], B=[0,0,0])
        self.router.channel_width = 6
        # Setup: split net 1 on tracks 2, 4
        self.router.Y = {1: set([2, 4])}
        self.router.collapse_split_nets()
        self.assertEqual(len(self.router.Y[1]), 1)

        # Test 1: one jog from 2 to 4, closes net
        self.router.reset(T=[1,0,0], B=[0,0,0])
        self.router.current_column = 1
        self.router.channel_width = 6
        # Setup: split net 1 on tracks 2, 4
        self.router.Y = {1: set([2, 4])}
        self.router.collapse_split_nets()
        self.assertEqual(len(self.router.Y[1]), 1)
        self.router.extend_nets() # should close net
        self.assertEqual(len(self.router.Y[1]), 0)

    def test_scout(self):
        # Jog up to reachable track
        self.router.reset(T=[0,0,1], B=[0,0,0])
        self.router.channel_width = 6
        self.router.Y = {1: set([2, 4])}
        # No vertical wiring, should reach goal
        result = self.router.scout(1, 2, 4)
        self.assertEqual(result, 4)

        # Jog down to reachable track
        result = self.router.scout(1, 4, 2)
        self.assertEqual(result, 2)

        # Jog blocked by vertical wire
        self.router.reset(T=[0,0,1], B=[0,0,99])
        self.router.channel_width = 6
        self.router.Y = {1: set([1, 4])}
        self.router.G.add_edge((0, 3), (0, 5), net=99)  # vertical wire at track 3
        # Should stop at 2 (can't reach 4)
        result = self.router.scout(1, 1, 4)
        self.assertEqual(result, 2)

        # No movement needed (already at goal)
        result = self.router.scout(1, 2, 2)
        self.assertEqual(result, 2)

    def test_compress_split_net(self):
        # Test 1: split net with a vertical wire in between, cannot be collapsed
        self.router.reset(T=[0,0,1], B=[0,0,0])
        self.router.channel_width = 6
        # Net 1 on tracks 1 and 4, net 2 on track 3, with a vertical wire
        self.router.Y = {1: set([1, 4]), 2: set([3])}
        self.router.add_vertical_wire(2, 3, 6)
        self.router.compress_split_net(1)
        # Net 1 should still be split (cannot collapse)
        self.assertEqual(len(self.router.Y[1]), 2)
        # But it should have been compressed
        self.assertEqual(self.router.Y[1], set([2, 4]))

        # Test 2: jog not long enough (minimum_jog_length > distance), cannot be collapsed
        self.router.reset(T=[0,0,1], B=[0,0,0], minimum_jog_length=3)
        self.router.channel_width = 6
        self.router.Y = {1: set([2, 4])}
        self.router.compress_split_net(1)
        # Net 1 should still be split 
        self.assertEqual(self.router.Y[1], set([2, 4]))
        self.assertEqual(len(self.router.Y[1]), 2)

    def test_push_unsplit_nets(self):
        # Rising net: should be pushed up to the top track
        self.router.reset(T=[0, 1, 0, 1], B=[0, 0, 0, 0])
        self.router.channel_width = 4
        self.router.Y = {1: set([2])}
        self.router.current_column = 1
        self.router.push_unsplit_nets()
        # Should be pushed to the top (track 4)
        self.assertIn(4, self.router.Y[1])

        # Falling net: should be pushed down to the bottom track
        self.router.reset(T=[0, 0, 0, 0], B=[0, 1, 0, 1])
        self.router.channel_width = 4
        self.router.Y = {1: set([3])}
        self.router.current_column = 1
        self.router.push_unsplit_nets()
        # Should be pushed to the bottom (track 1)
        self.assertIn(1, self.router.Y[1])

        # Steady net: should not be pushed
        self.router.reset(T=[0, 1, 0, 0], B=[0, 1, 0, 0])
        self.router.channel_width = 4
        self.router.Y = {1: set([2])}
        self.router.current_column = 1
        before = set(self.router.Y[1])
        self.router.push_unsplit_nets()
        # Should remain unchanged
        self.assertEqual(self.router.Y[1], before)

    def test_widen_channel(self):
        # Setup: tracks 2, 3, 4 occupied by net 1
        self.router.reset(T=[0, 0, 0], B=[0, 0, 0])
        self.router.channel_width = 4
        self.router.Y = {1: set([2, 3, 4])}
        # Add a node at (0, 3) and (0, 4) to test relabeling
        self.router.G.add_node((0, 3))
        self.router.G.add_node((0, 4))
        # Widen the channel
        self.router.widen_channel()
        # Channel width should increase by 1
        self.assertEqual(self.router.channel_width, 5)
        # Tracks at or above the new track (3) should be shifted up by 1
        # (default mid_track is 3)
        self.assertIn(5, self.router.Y[1])
        self.assertIn(4, self.router.Y[1])
        self.assertIn(2, self.router.Y[1])
        # Node at (0, 3) should now be at (0, 4), (0, 4) at (0, 5)
        self.assertIn((0, 4), self.router.G.nodes)
        self.assertIn((0, 5), self.router.G.nodes)
        # Track below new track (2) should remain unchanged
        self.assertIn(2, self.router.Y[1])

    def test_extend_nets(self):
        pass

    def test_route(self):
        pass

    def test_route_and_retry(self):
        pass

    def test_initialize_grid(self):
        pass

    def test__get_node_orientation(self):
        pass

    def test_render_text_graph(self):
        # Create snapshots for each test case
        snapshots = [
"""
         
1 ──╮    
    │    
    ╰── 1        
""",
"""
              
    ╭── 1
    │    
1 ──┴── 1        
""",
"""
    ╭── 1
1 ──┼── 1
    ╰── 1
""",
"""
        ╭── 2
1 ────╮ │    
    ╭─│─│── 3
2 ──│─│─╯    
3 ──╯ ╰──── 1
""",
"""
1 ────╮     ╭── 5
    ╭─│───╮ │    
2 ──│─│─╮ ╰─│── 4
    │ │ │   │    
3 ──│─│─│───│── 3
    │ │ │   │    
4 ──╯ │ ╰───│── 2
    ╭─│─────╯    
5 ──╯ ╰──────── 1

""",
"""
            ╭───╮        
        ╭───╯ ╭─│─╮      
    ╭───│─╮ ╭─│─│─│─╮    
1 ──│─╮ │ ╰─│─│─│─│─│── 8
2 ──│─│─│─╮ │ │ ╰─│─│── 7
3 ──│─│─│─│─│─│─╮ ╰─│── 6
4 ──│─│─│─│─╯ │ │ ╭─│── 5
5 ──│─│─│─│───│─│─╯ ╰── 4
6 ──│─│─│─│───╯ ╰────── 3
7 ──│─│─╯ ╰──────────── 2
8 ──╯ ╰──────────────── 1
""",
"""
3 ──┬────── 3
2 ──│─┬──── 2
1 ──│─│─┬── 1
3 ──┴─│─│── 3
2 ────┴─│── 2
1 ──────┴── 1
""",

        ]

        # Test cases
        test_cases = [
            ([0, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 0]),
            ([0, 1, 1, 1, 0], [0, 0, 1, 0, 0]),
            ([1, 0, 3, 0, 2], [3, 2, 0, 1, 0]),
            ([1, 0, 2, 0, 3, 0, 4, 0, 5], [5, 0, 4, 0, 3, 0, 2, 0, 1]),
            ([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]),
            ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]),
        ]

        # Run tests and compare with snapshots
        print("Testing render_text_graph:")
        for i, (T, B) in enumerate(test_cases):
            R = Router(T, B)
            Graph = R.route()
            P = Plotter(Graph)
            result = P.render_text_graph()
            print(result)
            print('='*60)
            # Compare with snapshot, normalizing strings to handle formatting differences
            result_normalized = self._normalize_string(result)
            snapshot_normalized = self._normalize_string(snapshots[i])
            
            self.assertEqual(result_normalized, snapshot_normalized, 
                            f"Test case {i+1} failed: output doesn't match snapshot")
    
    def _normalize_string(self, s):
        """Normalize a string for robust comparison by:
        1. Stripping leading/trailing whitespace from each line
        2. Removing empty lines
        3. Normalizing line endings
        """
        # Split the string into lines, normalize whitespace
        lines = s.splitlines()
        # Filter out empty lines and strip each line
        non_empty_lines = [line.rstrip() for line in lines if line.strip()]
        # Join back with newline and return
        return '\n'.join(non_empty_lines)
    
    def _random_pins(self,N, M):
        pins = list(range(N+1))
        pins.extend(random.choices(pins, k = M - N))
        random.shuffle(pins)
        # Add spacing between pins
        pins_spaced = []
        for pin in pins:
            pins_spaced.extend([pin, 0])
        pins_spaced.pop()
        return pins_spaced

    def test_random_graphs(self):
        # Set random seed for reproducibility
        random.seed(42)

        with open('edge_router_snapshot_test.txt', 'w') as f:
            print()
            for i in range(10):
                L = self._random_pins(10, 15)
                R = self._random_pins(10, 15)
                router = Router(L, R)
                G = router.route_and_retry()
                output = Plotter(G).render_text_graph()
                f.write(output)
                f.write('\n\n')

        # Assert that this file is the same as edge_router_snapshot_reference.txt
        with open('edge_router_snapshot_reference.txt', 'r') as f:
            reference = f.read()
        with open('edge_router_snapshot_test.txt', 'r') as f:
            test = f.read()
        self.assertEqual(reference, test)

if __name__ == '__main__':
    unittest.main() 
