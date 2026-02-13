use std::collections::VecDeque;

use log::warn;

pub struct MessageBuffer {
    entries: VecDeque<String>,
    capacity: usize,
}

impl MessageBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
        }
    }

    pub fn push_warn(&mut self, message: impl Into<String>) {
        let message = message.into();
        warn!("{message}");
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(message);
    }

    pub fn latest(&self) -> Option<&String> {
        self.entries.back()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.entries.iter()
    }
}
