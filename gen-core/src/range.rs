use std::cmp::{max, min};

use itertools::Itertools;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Range {
    pub start: i64,
    pub end: i64,
}

impl Range {
    pub fn extend_to(&self, other: &Range) -> Range {
        Range {
            start: self.start,
            end: other.end,
        }
    }

    // Returns true if this range is left-adjacent to the other range
    pub fn left_adjoins(&self, other: &Range, modulus: Option<i64>) -> bool {
        let mut other_start = other.start;
        let mut self_end = self.end;
        if let Some(modulus) = modulus {
            other_start %= modulus;
            self_end %= modulus;
        }

        self_end == other_start
    }

    pub fn is_wraparound(&self) -> bool {
        self.start > self.end
    }

    pub fn overlap(&self, other: &Range) -> Vec<Range> {
        /*
           Returns the overlapping ranges between two ranges. If there are multiple overlapping ranges,
           such as can be the case when a range wraps the origin, multiple ranges are returned. If
           there are no overlapping ranges, an empty list is returned.

           Examples:

           Overlap between two non-wraparound ranges
                 6            19
                 |------------|        self
           AAAAAAAAAAAAAAAAAAAAAAAA
           |------------|              other
           0            13
                 |------|              overlap
                 6      13

           Overlap with a wraparound range
             2   6
           >-|   |---------------->    self
           AAAAAAAAAAAAAAAAAAAAAAAA
           |---|                       other
           0   4
           |-|                         overlap
           0 2

           Overlap with multiple wraparound ranges
             2   6
           >-|   |---------------->    self
           AAAAAAAAAAAAAAAAAAAAAAAA
           >---|        |--------->    other
               4        13
           >-|          |--------->    overlap
             2          13

           Multiple Overlaps
               4        13
           >---|        |--------->    self
           AAAAAAAAAAAAAAAAAAAAAAAA
             |----------------|        other
             2                19
             |-|        |-----|        overlaps
             2 4        13    19
        */

        let start1 = self.start;
        let end1 = self.end;
        let start2 = other.start;
        let end2 = other.end;

        let mut self_intervals = vec![];
        let mut other_intervals = vec![];

        // split the ranges into pre-/post-origin segments
        if self.is_wraparound() {
            self_intervals.extend(vec![
                Range {
                    start: start1,
                    end: i64::MAX,
                },
                Range {
                    start: 1,
                    end: end1,
                },
            ]);
        } else {
            self_intervals.push(self.clone());
        }

        if other.is_wraparound() {
            other_intervals.extend(vec![
                Range {
                    start: start2,
                    end: i64::MAX,
                },
                Range {
                    start: 1,
                    end: end2,
                },
            ]);
        } else {
            other_intervals.push(other.clone());
        }

        let overlaps = Range::find_pairwise_overlaps(self_intervals, other_intervals);

        if overlaps.len() > 1 {
            Range::consolidate_overlaps_about_the_origin(overlaps)
        } else {
            overlaps
        }
    }

    fn find_pairwise_overlaps(intervals1: Vec<Range>, intervals2: Vec<Range>) -> Vec<Range> {
        let mut overlaps = vec![];
        for interval1 in intervals1 {
            for interval2 in &intervals2 {
                if interval1.end > interval2.start && interval1.start <= interval2.end {
                    overlaps.push(Range {
                        start: max(interval1.start, interval2.start),
                        end: min(interval1.end, interval2.end),
                    });
                }
            }
        }

        overlaps
    }

    // Consolidate the first and last overlaps if they lie on either side of the origin.
    fn consolidate_overlaps_about_the_origin(overlaps: Vec<Range>) -> Vec<Range> {
        let mut sorted_overlaps = overlaps
            .clone()
            .into_iter()
            .sorted_by(|a, b| a.start.cmp(&b.start))
            .collect::<Vec<Range>>();
        let first = sorted_overlaps.first().unwrap().clone();
        let last = sorted_overlaps.last().unwrap().clone();
        if first.start == 0 && last.end == i64::MAX {
            sorted_overlaps.pop();
            sorted_overlaps.push(Range {
                start: last.start,
                end: first.end,
            });
        }

        sorted_overlaps
    }

    pub fn contains(&self, index: i64) -> bool {
        if self.is_wraparound() {
            index >= self.start || index <= self.end
        } else {
            index >= self.start && index <= self.end
        }
    }

    pub fn circular_mod(index: i64, sequence_length: i64, is_circular_contig: bool) -> i64 {
        if is_circular_contig {
            index % sequence_length
        } else {
            index
        }
    }

    pub fn translate_index(
        &self,
        index: i64,
        range2: &Range,
        sequence_length: i64,
        is_circular_contig: bool,
    ) -> Result<i64, &'static str> {
        if !self.contains(index) {
            return Err("Index is not contained in range");
        }

        let offset = index - self.start;
        Ok(Range::circular_mod(
            range2.start + offset,
            sequence_length,
            is_circular_contig,
        ))
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RangeMapping {
    pub source_range: Range,
    pub target_range: Range,
}

impl RangeMapping {
    pub fn merge_contiguous_mappings(mappings: Vec<RangeMapping>) -> Vec<RangeMapping> {
        let mut grouped_mappings = vec![];
        let mut current_group = vec![];

        for mapping in mappings {
            if current_group.is_empty() {
                current_group.push(mapping);
            } else {
                let last_mapping = current_group.last().unwrap();
                if last_mapping
                    .source_range
                    .left_adjoins(&mapping.source_range, None)
                    && last_mapping
                        .target_range
                        .left_adjoins(&mapping.target_range, None)
                {
                    current_group.push(mapping);
                } else {
                    grouped_mappings.push(current_group);
                    current_group = vec![mapping];
                }
            }
        }

        if !current_group.is_empty() {
            grouped_mappings.push(current_group);
        }

        let mut merged_mappings = vec![];
        for group in grouped_mappings {
            let first = group.first().unwrap();
            let last = group.last().unwrap();
            merged_mappings.push(RangeMapping {
                source_range: first.source_range.extend_to(&last.source_range),
                target_range: first.target_range.extend_to(&last.target_range),
            });
        }

        merged_mappings
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_left_adjoins() {
        let left_range = Range { start: 0, end: 2 };
        let middle_range = Range { start: 1, end: 3 };
        let right_range = Range { start: 2, end: 4 };

        assert!(left_range.left_adjoins(&right_range, None));
        assert!(!left_range.left_adjoins(&middle_range, None));
        assert!(!middle_range.left_adjoins(&right_range, None));
        assert!(!right_range.left_adjoins(&left_range, None));
        assert!(!right_range.left_adjoins(&middle_range, None));
        assert!(!middle_range.left_adjoins(&left_range, None));

        assert!(right_range.left_adjoins(&left_range, Some(4)));
        assert!(left_range.left_adjoins(&right_range, Some(4)));
        assert!(!left_range.left_adjoins(&middle_range, Some(4)));
        assert!(!middle_range.left_adjoins(&right_range, Some(4)));
        assert!(!right_range.left_adjoins(&middle_range, Some(4)));
        assert!(!middle_range.left_adjoins(&left_range, Some(4)));
    }

    #[test]
    fn test_overlap() {
        let range1 = Range { start: 0, end: 12 };
        let range2 = Range { start: 4, end: 8 };
        let range3 = Range { start: 10, end: 16 };

        assert_eq!(range1.overlap(&range2), vec![Range { start: 4, end: 8 }]);
        assert_eq!(range1.overlap(&range3), vec![Range { start: 10, end: 12 }]);
        assert_eq!(range2.overlap(&range3), vec![]);
    }

    #[test]
    fn test_merge_contiguous_ranges() {
        let mappings = vec![
            RangeMapping {
                source_range: Range { start: 0, end: 2 },
                target_range: Range { start: 2, end: 4 },
            },
            RangeMapping {
                source_range: Range { start: 2, end: 5 },
                target_range: Range { start: 4, end: 7 },
            },
            RangeMapping {
                source_range: Range { start: 7, end: 8 },
                target_range: Range { start: 9, end: 10 },
            },
        ];

        let merged_mappings = RangeMapping::merge_contiguous_mappings(mappings);
        assert_eq!(merged_mappings.len(), 2);
        assert_eq!(
            merged_mappings,
            vec![
                RangeMapping {
                    source_range: Range { start: 0, end: 5 },
                    target_range: Range { start: 2, end: 7 },
                },
                RangeMapping {
                    source_range: Range { start: 7, end: 8 },
                    target_range: Range { start: 9, end: 10 },
                },
            ]
        );
    }
}
