//! Progress hooks used while preparing annotation files.

use std::io::{self, Write};

use super::AnnotationFileChecksumOverrides;

/// A stage reported while preparing a local GFF3 or BED annotation file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnnotationFileProgressStage {
    /// Sort source records before creating offset-based indexes.
    Sorting,
    /// Compress the sorted source into the retained BGZF asset.
    Compressing,
    /// Build the searchable attribute index for the retained source.
    BuildingIndex,
}

/// Options shared by annotation-file addition and its optional progress callback.
pub struct AnnotationFileAddOptions<'a> {
    /// Checksums collected while reading annotation assets.
    pub checksum_overrides: AnnotationFileChecksumOverrides,
    /// Called immediately before each local GFF3 or BED preparation stage.
    pub progress: Option<&'a mut dyn FnMut(AnnotationFileProgressStage)>,
}

pub(super) fn report_annotation_file_progress(
    progress: &mut Option<&mut dyn FnMut(AnnotationFileProgressStage)>,
    stage: AnnotationFileProgressStage,
) {
    if let Some(callback) = progress.as_mut() {
        callback(stage);
    }
}

// GAI sorts all records before it emits sorted rows, so the first BGZF write marks the transition.
pub(super) struct ReportCompressionOnFirstWrite<W, F> {
    writer: W,
    on_first_write: Option<F>,
}

impl<W, F> ReportCompressionOnFirstWrite<W, F> {
    pub(super) fn new(writer: W, on_first_write: F) -> Self {
        Self {
            writer,
            on_first_write: Some(on_first_write),
        }
    }
}

impl<W: Write, F: FnMut()> Write for ReportCompressionOnFirstWrite<W, F> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if !buffer.is_empty()
            && let Some(mut on_first_write) = self.on_first_write.take()
        {
            on_first_write();
        }

        self.writer.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
