use std::cmp::{max, min};

use itertools::Itertools;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
            self_intervals.push(*self);
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
            other_intervals.push(*other);
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
        let first = *sorted_overlaps.first().unwrap();
        let last = *sorted_overlaps.last().unwrap();
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

/// Merges an already ordered stream of items where each item decides whether it can merge with
/// the current tail and how that merge updates the tail in place.
pub trait OrderedMerge: Sized {
    fn should_merge_with(&self, next: &Self) -> bool;
    fn merge_with(&mut self, next: &Self);
}

pub fn merge_ordered_items<T: OrderedMerge>(items: Vec<T>) -> Vec<T> {
    let mut merged: Vec<T> = Vec::with_capacity(items.len());

    for item in items {
        if let Some(last) = merged.last_mut()
            && last.should_merge_with(&item)
        {
            last.merge_with(&item);
        } else {
            merged.push(item);
        }
    }

    merged
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RangeMapping {
    pub source_range: Range,
    pub target_range: Range,
}

impl OrderedMerge for RangeMapping {
    fn should_merge_with(&self, next: &Self) -> bool {
        self.source_range.left_adjoins(&next.source_range, None)
            && self.target_range.left_adjoins(&next.target_range, None)
    }

    fn merge_with(&mut self, next: &Self) {
        self.source_range = self.source_range.extend_to(&next.source_range);
        self.target_range = self.target_range.extend_to(&next.target_range);
    }
}

impl RangeMapping {
    pub fn merge_contiguous_mappings(mappings: Vec<RangeMapping>) -> Vec<RangeMapping> {
        merge_ordered_items(mappings)
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
