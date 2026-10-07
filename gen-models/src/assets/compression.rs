use std::{
    fs,
    io::{self, Chain, Cursor, Read, Write},
};

use gen_core::{Sha256Hash, Workspace};
use noodles::bgzf;
use tempfile::NamedTempFile;

use super::{ChecksummedReader, ChecksummedWriter, FileTypes};
use crate::errors::FileAdditionError;

/// A compression encoding inferred from an asset path or detected from its bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressionType {
    /// The bytes are uncompressed or their path does not identify a compressed format.
    Plain,
    /// The bytes use ordinary gzip compression.
    Gzip,
    /// The bytes use BGZF compression, including BAM files.
    Bgzf,
}

type ReplayedReader<R> = Chain<Cursor<Vec<u8>>, R>;

impl CompressionType {
    /// Classifies input compression from header bytes and returns a reader that replays inspected bytes.
    ///
    /// Replaying lets callers hash or decode the complete stream without losing bytes read during
    /// classification. [`super::AssetUri::compression_type`] infers from the URI suffix only.
    pub fn sniff<R: Read>(reader: R) -> io::Result<(Self, impl Read)> {
        classify_input(reader)
    }
}

// Read enough header bytes to distinguish BGZF from ordinary gzip, then replay them into storage.
fn classify_input<R: Read>(mut reader: R) -> io::Result<(CompressionType, ReplayedReader<R>)> {
    let mut prefix = Vec::with_capacity(12);
    let encoding = if !fill_prefix(&mut reader, &mut prefix, 2)? || prefix[..2] != [0x1f, 0x8b] {
        CompressionType::Plain
    } else if !fill_prefix(&mut reader, &mut prefix, 10)?
        || prefix[2] != 8
        || prefix[3] & 0x04 == 0
        || !fill_prefix(&mut reader, &mut prefix, 12)?
    {
        CompressionType::Gzip
    } else {
        let extra_length = usize::from(u16::from_le_bytes([prefix[10], prefix[11]]));
        if !fill_prefix(&mut reader, &mut prefix, 12 + extra_length)? {
            CompressionType::Gzip
        } else if has_bgzf_subfield(&prefix[12..]) {
            CompressionType::Bgzf
        } else {
            CompressionType::Gzip
        }
    };

    Ok((encoding, Cursor::new(prefix).chain(reader)))
}

// This creates a temporary bgzipped file that is then copied into the asset directory.
//
// It creates a checksum as the file is read and compressed. If the incoming file is already
// compressed with bgz, it copied through. If a file is compressed already with an algorithm
// such as gzip, it is decompressed and then recompressed as bgz. This is because many places will
// ship a fasta or other assets as a .gz which does not allow random access.
pub(crate) fn stage_bgzf_asset_copy(
    workspace: &Workspace,
    file_type: FileTypes,
    reader: impl Read + 'static,
    compression_type: CompressionType,
    source_checksum_override: Option<Sha256Hash>,
) -> Result<(Sha256Hash, Sha256Hash), FileAdditionError> {
    let asset_dir = workspace.asset_dir()?;
    fs::create_dir_all(&asset_dir).map_err(FileAdditionError::FileReadError)?;

    let source_reader = ChecksummedReader::new(reader);
    let source_checksum_handle = source_reader.checksum_handle();
    let mut staged_file =
        NamedTempFile::new_in(&asset_dir).map_err(FileAdditionError::FileReadError)?;
    let archive_checksum = {
        let checksummed_writer = ChecksummedWriter::new(staged_file.as_file_mut());
        match compression_type {
            CompressionType::Gzip => {
                let mut bgzf_writer = bgzf::io::Writer::new(checksummed_writer);
                io::copy(
                    &mut flate2::read::MultiGzDecoder::new(source_reader),
                    &mut bgzf_writer,
                )
                .map_err(FileAdditionError::FileReadError)?;
                let mut checksummed_writer = bgzf_writer
                    .finish()
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer
                    .flush()
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer.checksum()
            }
            CompressionType::Bgzf => {
                let mut checksummed_writer = checksummed_writer;
                let mut source_reader = source_reader;
                io::copy(&mut source_reader, &mut checksummed_writer)
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer
                    .flush()
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer.checksum()
            }
            CompressionType::Plain => {
                let mut bgzf_writer = bgzf::io::Writer::new(checksummed_writer);
                let mut source_reader = source_reader;
                io::copy(&mut source_reader, &mut bgzf_writer)
                    .map_err(FileAdditionError::FileReadError)?;
                let mut checksummed_writer = bgzf_writer
                    .finish()
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer
                    .flush()
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer.checksum()
            }
        }
    };
    staged_file
        .flush()
        .map_err(FileAdditionError::FileReadError)?;
    let source_checksum = source_checksum_handle.checksum().ok_or_else(|| {
        FileAdditionError::ChecksumError("local asset stream did not reach EOF".to_string())
    })?;
    if let Some(expected_checksum) = source_checksum_override
        && source_checksum != expected_checksum
    {
        return Err(FileAdditionError::ChecksumError(
            "local source checksum does not match the provided checksum".to_string(),
        ));
    }

    let archive_filename = format!("{archive_checksum}.{}.bgz", FileTypes::suffix(file_type));
    let archived_path = asset_dir.join(archive_filename);
    match staged_file.persist_noclobber(&archived_path) {
        Ok(_) => {}
        Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(FileAdditionError::FileReadError(error.error)),
    }
    Ok((archive_checksum, source_checksum))
}

fn fill_prefix<R: Read>(reader: &mut R, prefix: &mut Vec<u8>, target: usize) -> io::Result<bool> {
    let mut buffer = [0; 8192];
    while prefix.len() < target {
        let remaining = target - prefix.len();
        let read_length = remaining.min(buffer.len());
        let read = match reader.read(&mut buffer[..read_length]) {
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if read == 0 {
            return Ok(false);
        }
        prefix.extend_from_slice(&buffer[..read]);
    }
    Ok(true)
}

fn has_bgzf_subfield(extra: &[u8]) -> bool {
    let mut offset = 0;
    while offset + 4 <= extra.len() {
        let subfield_length =
            usize::from(u16::from_le_bytes([extra[offset + 2], extra[offset + 3]]));
        let end = offset + 4 + subfield_length;
        if end > extra.len() {
            return false;
        }
        if extra[offset] == b'B' && extra[offset + 1] == b'C' && subfield_length == 2 {
            return true;
        }
        offset = end;
    }
    false
}

#[cfg(test)]
mod tests {
    use std::io::{self, Cursor, Read, Write as _};

    use noodles::bgzf;

    use super::CompressionType;

    struct ShortReadReader<R> {
        reader: R,
        maximum_read: usize,
    }

    impl<R: Read> Read for ShortReadReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let maximum_read = buffer.len().min(self.maximum_read);
            self.reader.read(&mut buffer[..maximum_read])
        }
    }

    #[test]
    fn test_classify_replays_short_read_plain_gzip_and_bgzf_inputs() {
        let mut gzip_writer =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip_writer
            .write_all(b"plain bytes")
            .expect("should write gzip bytes");
        let gzip_contents = gzip_writer.finish().expect("should finish gzip bytes");
        let mut bgzf_writer = bgzf::io::Writer::new(Vec::new());
        bgzf_writer
            .write_all(b"bgzf bytes")
            .expect("should write BGZF bytes");
        let bgzf_contents = bgzf_writer.finish().expect("should finish BGZF bytes");
        let cases = [
            (b"plain bytes".to_vec(), CompressionType::Plain),
            (gzip_contents, CompressionType::Gzip),
            (bgzf_contents, CompressionType::Bgzf),
        ];

        for (contents, expected_encoding) in cases {
            let reader = ShortReadReader {
                reader: Cursor::new(contents.clone()),
                maximum_read: 1,
            };
            let (encoding, mut replayed_reader) =
                CompressionType::sniff(reader).expect("should classify source bytes");
            let mut replayed_contents = Vec::new();
            replayed_reader
                .read_to_end(&mut replayed_contents)
                .expect("should replay the complete source");

            assert_eq!(encoding, expected_encoding);
            assert_eq!(replayed_contents, contents);
        }
    }

    #[test]
    fn test_classify_finds_bgzf_after_a_full_gzip_extra_field() {
        let mut contents = vec![0x1f, 0x8b, 8, 0x04, 0, 0, 0, 0, 0, 255];
        contents.extend_from_slice(&u16::MAX.to_le_bytes());
        let unknown_subfield_length = usize::from(u16::MAX) - 6;
        let unknown_payload_length = unknown_subfield_length - 4;
        contents.extend_from_slice(&[
            b'X',
            b'Y',
            unknown_payload_length as u8,
            (unknown_payload_length >> 8) as u8,
        ]);
        contents.resize(12 + unknown_subfield_length, 0);
        contents.extend_from_slice(&[b'B', b'C', 2, 0, 0, 0]);
        contents.extend_from_slice(b"compressed payload");

        let reader = ShortReadReader {
            reader: Cursor::new(contents.clone()),
            maximum_read: 7,
        };
        let (encoding, mut replayed_reader) =
            CompressionType::sniff(reader).expect("should parse the complete gzip extra field");
        let mut replayed_contents = Vec::new();
        replayed_reader
            .read_to_end(&mut replayed_contents)
            .expect("should replay all source bytes");

        assert_eq!(encoding, CompressionType::Bgzf);
        assert_eq!(replayed_contents, contents);
    }

    mod stage_bgzf_asset_copy_tests {
        use std::{
            fs::{self, File, FileTimes},
            io::{Cursor, Read, Write as _},
            time::{Duration, SystemTime},
        };

        use gen_core::{Sha256Hash, Workspace};
        use noodles::bgzf;
        use sha2::{Digest, Sha256};
        use tempfile::{TempDir, tempdir};

        use super::super::{CompressionType, FileTypes, stage_bgzf_asset_copy};

        fn setup_workspace(temp_dir: &TempDir) -> Workspace {
            let workspace = Workspace::new(temp_dir.path());
            workspace.ensure_gen_dir();
            workspace
        }

        fn checksum(bytes: &[u8]) -> Sha256Hash {
            Sha256Hash(Sha256::digest(bytes).into())
        }

        fn retained_archive_path(
            workspace: &Workspace,
            archive_checksum: Sha256Hash,
        ) -> std::path::PathBuf {
            workspace
                .asset_dir()
                .expect("should find asset directory")
                .join(format!(
                    "{archive_checksum}.{}.bgz",
                    FileTypes::suffix(FileTypes::Fasta)
                ))
        }

        fn assert_retained_bgzf_asset(
            workspace: &Workspace,
            archive_checksum: Sha256Hash,
            expected_contents: &[u8],
        ) -> Vec<u8> {
            let archive_path = retained_archive_path(workspace, archive_checksum);
            let archive_bytes = fs::read(archive_path).expect("should read retained BGZF asset");
            assert_eq!(
                checksum(&archive_bytes),
                archive_checksum,
                "archive checksum should match retained BGZF bytes"
            );
            let (compression_type, _) = CompressionType::sniff(Cursor::new(archive_bytes.clone()))
                .expect("should classify retained BGZF asset");
            assert_eq!(
                compression_type,
                CompressionType::Bgzf,
                "retained archive should use BGZF compression"
            );
            let mut decoder = bgzf::io::Reader::new(Cursor::new(archive_bytes.clone()));
            let mut decoded_contents = Vec::new();
            decoder
                .read_to_end(&mut decoded_contents)
                .expect("should decode retained BGZF asset");
            assert_eq!(
                decoded_contents, expected_contents,
                "retained BGZF asset should decode to source contents"
            );
            archive_bytes
        }

        #[test]
        fn test_stage_bgzf_asset_copy_compresses_gzip_input() {
            let temp_dir = tempdir().expect("should create temporary directory");
            let workspace = setup_workspace(&temp_dir);
            let expected_contents = b">sequence\nACGT\n";
            let mut gzip_writer =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            gzip_writer
                .write_all(expected_contents)
                .expect("should write gzip source");
            let source_bytes = gzip_writer.finish().expect("should finish gzip source");

            let (archive_checksum, source_checksum) = stage_bgzf_asset_copy(
                &workspace,
                FileTypes::Fasta,
                Cursor::new(source_bytes.clone()),
                CompressionType::Gzip,
                None,
            )
            .expect("should retain gzip input as BGZF");

            assert_eq!(
                source_checksum,
                checksum(&source_bytes),
                "source checksum should cover the exact gzip input bytes"
            );
            assert_retained_bgzf_asset(&workspace, archive_checksum, expected_contents);
        }

        #[test]
        fn test_stage_bgzf_asset_copy_retains_bgzf_input() {
            let temp_dir = tempdir().expect("should create temporary directory");
            let workspace = setup_workspace(&temp_dir);
            let expected_contents = b">sequence\nACGT\n";
            let mut bgzf_writer = bgzf::io::writer::Builder::default()
                .set_compression_level(bgzf::io::writer::CompressionLevel::NONE)
                .build_from_writer(Vec::new());
            bgzf_writer
                .write_all(expected_contents)
                .expect("should write BGZF source");
            let source_bytes = bgzf_writer.finish().expect("should finish BGZF source");

            let (archive_checksum, source_checksum) = stage_bgzf_asset_copy(
                &workspace,
                FileTypes::Fasta,
                Cursor::new(source_bytes.clone()),
                CompressionType::Bgzf,
                None,
            )
            .expect("should retain BGZF input");

            let expected_checksum = checksum(&source_bytes);
            assert_eq!(source_checksum, expected_checksum);
            assert_eq!(archive_checksum, expected_checksum);
            assert_eq!(
                assert_retained_bgzf_asset(&workspace, archive_checksum, expected_contents),
                source_bytes,
                "retained BGZF bytes should be identical to source bytes"
            );
        }

        #[test]
        fn test_stage_bgzf_asset_copy_compresses_plain_input() {
            let temp_dir = tempdir().expect("should create temporary directory");
            let workspace = setup_workspace(&temp_dir);
            let source_bytes = b">sequence\nACGT\n";

            let (archive_checksum, source_checksum) = stage_bgzf_asset_copy(
                &workspace,
                FileTypes::Fasta,
                Cursor::new(source_bytes.to_vec()),
                CompressionType::Plain,
                None,
            )
            .expect("should retain plain input as BGZF");

            assert_eq!(
                source_checksum,
                checksum(source_bytes),
                "source checksum should cover the exact plain input bytes"
            );
            assert_retained_bgzf_asset(&workspace, archive_checksum, source_bytes);
        }

        #[test]
        fn test_stage_bgzf_asset_copy_does_not_overwrite_existing_archive() {
            let temp_dir = tempdir().expect("should create temporary directory");
            let workspace = setup_workspace(&temp_dir);
            let mut bgzf_writer = bgzf::io::Writer::new(Vec::new());
            bgzf_writer
                .write_all(b">sequence\nACGT\n")
                .expect("should write BGZF source");
            let source_bytes = bgzf_writer.finish().expect("should finish BGZF source");
            let first_result = stage_bgzf_asset_copy(
                &workspace,
                FileTypes::Fasta,
                Cursor::new(source_bytes.clone()),
                CompressionType::Bgzf,
                None,
            )
            .expect("should retain first BGZF source");
            let archive_path = retained_archive_path(&workspace, first_result.0);
            let original_archive_bytes =
                fs::read(&archive_path).expect("should read first retained BGZF asset");

            let fixed_modified_time = SystemTime::UNIX_EPOCH + Duration::from_secs(1_600_000_000);
            File::options()
                .write(true)
                .open(&archive_path)
                .expect("should open retained BGZF asset")
                .set_times(FileTimes::new().set_modified(fixed_modified_time))
                .expect("should set retained asset modification time");
            let modified_time_before = fs::metadata(&archive_path)
                .expect("should inspect retained BGZF asset")
                .modified()
                .expect("should read retained asset modification time");

            let second_result = stage_bgzf_asset_copy(
                &workspace,
                FileTypes::Fasta,
                Cursor::new(source_bytes.clone()),
                CompressionType::Bgzf,
                None,
            )
            .expect("should retain duplicate BGZF source");

            assert_eq!(first_result, second_result);
            assert_eq!(
                fs::read(&archive_path).expect("should read retained BGZF asset again"),
                original_archive_bytes,
                "existing archive bytes should be unchanged"
            );
            assert_eq!(
                fs::metadata(&archive_path)
                    .expect("should inspect retained BGZF asset again")
                    .modified()
                    .expect("should read retained asset modification time again"),
                modified_time_before,
                "existing archive modification time should be unchanged"
            );
            assert_eq!(
                fs::read_dir(workspace.asset_dir().expect("should find asset directory"))
                    .expect("should read asset directory")
                    .count(),
                1,
                "duplicate retention should leave no staged files behind"
            );
        }
    }
}
