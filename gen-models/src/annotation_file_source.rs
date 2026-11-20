//! Coordinate provenance for file annotations, independent of the graph being viewed.

use thiserror::Error;

use crate::{
    collection::Collection,
    db::{DbContext, GraphConnection},
    operations::Defaults,
    sample::Sample,
};

/// Sample whose current block-group paths define an annotation file's coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationFileSource {
    /// Collection containing the source graphs.
    pub collection_name: String,
    /// Sample whose current paths define the file's coordinate system.
    pub sample_name: String,
}

#[derive(Debug, Error)]
pub enum AnnotationFileSourceError {
    #[error("Multiple collections are available; specify the annotation's collection")]
    AmbiguousCollection,
    #[error("Multiple reference samples exist in collection {0}; specify --sample")]
    AmbiguousSample(String),
    #[error("Unable to load annotation source: {0}")]
    Query(#[from] crate::ModelSelectError),
}

impl AnnotationFileSource {
    /// Resolves defaults when adding an annotation file, retaining explicit source names.
    pub fn for_addition(
        context: &DbContext,
        collection: Option<&str>,
        sample: Option<&str>,
    ) -> Result<Self, AnnotationFileSourceError> {
        let configured =
            Defaults::get(context.config().conn()).and_then(|defaults| defaults.collection_name);
        let collection = collection.or(configured.as_deref());
        let collection = match collection {
            Some(collection) => collection.to_string(),
            None => {
                let collections = Collection::select(context.graph().conn()).load()?;
                match collections.as_slice() {
                    [] => "default".to_string(),
                    [collection] => collection.name.clone(),
                    _ => return Err(AnnotationFileSourceError::AmbiguousCollection),
                }
            }
        };
        Self::resolve(context.graph().conn(), &collection, sample, None)
    }

    /// Infers a reference sample only when the caller has not selected a source sample.
    pub fn resolve(
        conn: &GraphConnection,
        collection: &str,
        sample: Option<&str>,
        history_ref: Option<&str>,
    ) -> Result<Self, AnnotationFileSourceError> {
        let sample_name = match sample {
            Some(sample) => sample.to_string(),
            None => {
                let references = Sample::get_reference_samples(conn, history_ref)
                    .into_iter()
                    .filter(|sample| {
                        !Sample::get_block_groups(conn, collection, &sample.name, history_ref)
                            .is_empty()
                    })
                    .collect::<Vec<_>>();
                match references.as_slice() {
                    [] => Sample::DEFAULT_NAME.to_string(),
                    [sample] => sample.name.clone(),
                    _ => {
                        return Err(AnnotationFileSourceError::AmbiguousSample(
                            collection.to_string(),
                        ));
                    }
                }
            }
        };
        Ok(Self {
            collection_name: collection.to_string(),
            sample_name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{AnnotationFileSource, AnnotationFileSourceError};
    use crate::{
        block_group::{BlockGroup, NewBlockGroup},
        collection::Collection,
        sample::{NewSample, Sample},
        test_helpers::setup_gen,
    };

    #[test]
    fn test_annotation_source_defaults_and_explicit_selection() {
        let context = setup_gen();
        let conn = context.graph().conn();
        Collection::create(conn, "genome").unwrap();
        assert_eq!(
            AnnotationFileSource::for_addition(&context, None, None).unwrap(),
            AnnotationFileSource {
                collection_name: "genome".to_string(),
                sample_name: Sample::DEFAULT_NAME.to_string(),
            }
        );
        for sample in ["assembly-one", "assembly-two"] {
            Sample::get_or_create(
                conn,
                NewSample {
                    name: sample,
                    is_reference: true,
                },
            )
            .unwrap();
            BlockGroup::create(
                conn,
                NewBlockGroup {
                    collection_name: "genome",
                    sample_name: sample,
                    name: "chr1",
                    ..Default::default()
                },
            )
            .unwrap();
            if sample == "assembly-one" {
                assert_eq!(
                    AnnotationFileSource::for_addition(&context, None, None)
                        .unwrap()
                        .sample_name,
                    sample
                );
            }
        }
        assert!(matches!(
            AnnotationFileSource::for_addition(&context, None, None),
            Err(AnnotationFileSourceError::AmbiguousSample(_))
        ));
        assert_eq!(
            AnnotationFileSource::for_addition(&context, None, Some("edited-sample"))
                .unwrap()
                .sample_name,
            "edited-sample"
        );
        Collection::create(conn, "another-genome").unwrap();
        assert!(matches!(
            AnnotationFileSource::for_addition(&context, None, Some("edited-sample")),
            Err(AnnotationFileSourceError::AmbiguousCollection)
        ));
        assert_eq!(
            AnnotationFileSource::for_addition(&context, Some("genome"), Some("edited-sample"))
                .unwrap()
                .collection_name,
            "genome"
        );
    }
}
