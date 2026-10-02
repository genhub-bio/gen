use std::{
    fs,
    io::{self, Chain, Cursor, Read, Write},
};

use gen_core::{Sha256Hash, Workspace};
use noodles::bgzf;
use tempfile::NamedTempFile;

use super::{ChecksummedReader, ChecksummedWriter, FileTypes, LocalAssetUri};
use crate::errors::FileAdditionError;

/// Returns whether a file type is retained as BGZF for efficient indexed access.
pub fn should_archive_as_bgzf(file_type: FileTypes) -> bool {
    matches!(
        file_type,
        FileTypes::Fasta
            | FileTypes::VCF
            | FileTypes::GFA
            | FileTypes::GAF
            | FileTypes::Gff3
            | FileTypes::Bed
            | FileTypes::GenBank
            | FileTypes::CSV
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InputEncoding {
    Plain,
    Gzip,
    Bgzf,
}

pub(crate) type ReplayedReader<R> = Chain<Cursor<Vec<u8>>, R>;

// Read enough header bytes to distinguish BGZF from ordinary gzip, then replay them into storage.
pub(crate) fn classify_input<R: Read>(
    mut reader: R,
) -> io::Result<(InputEncoding, ReplayedReader<R>)> {
    let mut prefix = Vec::with_capacity(12);
    let encoding = if !fill_prefix(&mut reader, &mut prefix, 2)? || prefix[..2] != [0x1f, 0x8b] {
        InputEncoding::Plain
    } else if !fill_prefix(&mut reader, &mut prefix, 10)?
        || prefix[2] != 8
        || prefix[3] & 0x04 == 0
        || !fill_prefix(&mut reader, &mut prefix, 12)?
    {
        InputEncoding::Gzip
    } else {
        let extra_length = usize::from(u16::from_le_bytes([prefix[10], prefix[11]]));
        if !fill_prefix(&mut reader, &mut prefix, 12 + extra_length)? {
            InputEncoding::Gzip
        } else if has_bgzf_subfield(&prefix[12..]) {
            InputEncoding::Bgzf
        } else {
            InputEncoding::Gzip
        }
    };

    Ok((encoding, Cursor::new(prefix).chain(reader)))
}

pub(crate) fn stage_bgzf_asset_copy(
    workspace: &Workspace,
    source_uri: &str,
    file_type: FileTypes,
    reader: impl Read + 'static,
    input_encoding: InputEncoding,
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
        match input_encoding {
            InputEncoding::Gzip => {
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
            InputEncoding::Bgzf => {
                let mut checksummed_writer = checksummed_writer;
                let mut source_reader = source_reader;
                io::copy(&mut source_reader, &mut checksummed_writer)
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer
                    .flush()
                    .map_err(FileAdditionError::FileReadError)?;
                checksummed_writer.checksum()
            }
            InputEncoding::Plain => {
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
        FileAdditionError::ChecksumError(format!(
            "local asset stream did not reach EOF: {source_uri}"
        ))
    })?;
    if let Some(expected_checksum) = source_checksum_override
        && source_checksum != expected_checksum
    {
        return Err(FileAdditionError::ChecksumError(format!(
            "local source checksum does not match the provided checksum: {source_uri}"
        )));
    }

    let archive_filename = format!("{archive_checksum}.{}.bgz", FileTypes::suffix(file_type));
    let archived_path = asset_dir.join(archive_filename);
    match staged_file.persist_noclobber(&archived_path) {
        Ok(_) => {}
        Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
            LocalAssetUri::verify_asset_checksum(&archived_path, archive_checksum, source_uri)?;
        }
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

    use super::{InputEncoding, classify_input};

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
            (b"plain bytes".to_vec(), InputEncoding::Plain),
            (gzip_contents, InputEncoding::Gzip),
            (bgzf_contents, InputEncoding::Bgzf),
        ];

        for (contents, expected_encoding) in cases {
            let reader = ShortReadReader {
                reader: Cursor::new(contents.clone()),
                maximum_read: 1,
            };
            let (encoding, mut replayed_reader) =
                classify_input(reader).expect("should classify source bytes");
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
            classify_input(reader).expect("should parse the complete gzip extra field");
        let mut replayed_contents = Vec::new();
        replayed_reader
            .read_to_end(&mut replayed_contents)
            .expect("should replay all source bytes");

        assert_eq!(encoding, InputEncoding::Bgzf);
        assert_eq!(replayed_contents, contents);
    }
}
