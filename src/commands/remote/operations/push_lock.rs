//! Process ownership for a push's persisted remote-operation session.

use std::{
    fs::{self, File, OpenOptions, TryLockError},
    path::{Path, PathBuf},
};

use gen_core::config::Workspace;
use sha2::{Digest as _, Sha256};

use super::RemotePushError;

/// Acquires the workspace lock that protects one remote and branch's resumable push session.
///
/// The open file is the guard: normal returns and errors drop it, while the OS releases it when a
/// process terminates. Coordination on a NAS depends on its filesystem and server honoring file
/// locks. The persistent file stays in `.gen` so different mount paths to one workspace address
/// the same lock object.
pub(super) fn acquire(
    workspace: &Workspace,
    remote_name: &str,
    branch_name: &str,
) -> Result<File, RemotePushError> {
    let gen_dir = workspace
        .find_gen_dir()
        .ok_or(gen_core::errors::ConfigError::GenDirectoryNotFound)?;
    let lock_dir = gen_dir.join("locks").join("push");
    fs::create_dir_all(&lock_dir).map_err(|source| RemotePushError::PushLockIo {
        remote: remote_name.to_string(),
        branch: branch_name.to_string(),
        path: lock_dir.clone(),
        source,
    })?;

    let path = lock_file_path(&gen_dir, remote_name, branch_name);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|source| RemotePushError::PushLockIo {
            remote: remote_name.to_string(),
            branch: branch_name.to_string(),
            path: path.clone(),
            source,
        })?;

    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(RemotePushError::PushLockBusy {
            remote: remote_name.to_string(),
            branch: branch_name.to_string(),
        }),
        Err(TryLockError::Error(source)) => Err(RemotePushError::PushLockIo {
            remote: remote_name.to_string(),
            branch: branch_name.to_string(),
            path,
            source,
        }),
    }
}

fn lock_file_path(gen_dir: &Path, remote_name: &str, branch_name: &str) -> PathBuf {
    let mut digest = Sha256::new();
    digest.update(b"gen/push-lock/v1\0");
    update_hash_field(&mut digest, remote_name.as_bytes());
    update_hash_field(&mut digest, branch_name.as_bytes());
    gen_dir
        .join("locks")
        .join("push")
        .join(format!("{:x}.lock", digest.finalize()))
}

fn update_hash_field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value);
}

#[cfg(test)]
pub(super) mod tests {
    use std::{
        env, fs,
        io::Write as _,
        path::Path,
        process::{Child, ChildStdin, Command, Stdio},
        thread,
        time::{Duration, Instant},
    };

    use gen_core::config::Workspace;
    use tempfile::tempdir;

    use super::{acquire, lock_file_path};
    use crate::commands::remote::operations::RemotePushError;

    const CHILD_ROOT_ENV: &str = "GEN_PUSH_LOCK_TEST_ROOT";
    const CHILD_REMOTE_ENV: &str = "GEN_PUSH_LOCK_TEST_REMOTE";
    const CHILD_BRANCH_ENV: &str = "GEN_PUSH_LOCK_TEST_BRANCH";
    const CHILD_READY_ENV: &str = "GEN_PUSH_LOCK_TEST_READY";

    pub(in crate::commands::remote::operations) struct LockHolder {
        child: Child,
        input: Option<ChildStdin>,
        finished: bool,
    }

    impl LockHolder {
        pub(in crate::commands::remote::operations) fn spawn(
            root: &Path,
            remote_name: &str,
            branch_name: &str,
        ) -> Self {
            let ready_path = root.join("push-lock-child.ready");
            let executable = env::current_exe().expect("should locate the current test binary");
            let mut child = Command::new(executable)
                .args([
                    "--exact",
                    "commands::remote::operations::push_lock::tests::test_lock_child_process_helper",
                    "--nocapture",
                ])
                .env(CHILD_ROOT_ENV, root)
                .env(CHILD_REMOTE_ENV, remote_name)
                .env(CHILD_BRANCH_ENV, branch_name)
                .env(CHILD_READY_ENV, &ready_path)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("should start the child lock holder");
            let input = child
                .stdin
                .take()
                .expect("should pipe child lock holder input");
            let mut holder = Self {
                child,
                input: Some(input),
                finished: false,
            };
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                if ready_path.exists() {
                    return holder;
                }
                if let Some(status) = holder
                    .child
                    .try_wait()
                    .expect("should check child lock holder status")
                {
                    panic!("child lock holder exited before acquiring its lock: {status}");
                }
                thread::sleep(Duration::from_millis(10));
            }
            panic!("child lock holder did not acquire its lock before timeout");
        }

        pub(in crate::commands::remote::operations) fn release(mut self) {
            self.input
                .take()
                .expect("should retain child lock holder input")
                .write_all(b"release\n")
                .expect("should release the child lock holder");
            let status = self
                .child
                .wait()
                .expect("should wait for the child lock holder");
            self.finished = true;
            assert!(status.success(), "child lock holder should exit cleanly");
        }

        fn terminate(mut self) {
            self.child
                .kill()
                .expect("should terminate the child lock holder");
            self.child
                .wait()
                .expect("should reap the terminated child lock holder");
            self.finished = true;
        }
    }

    impl Drop for LockHolder {
        fn drop(&mut self) {
            if !self.finished {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    #[test]
    fn test_lock_child_process_helper() {
        let (Ok(root), Ok(remote_name), Ok(branch_name), Ok(ready_path)) = (
            env::var(CHILD_ROOT_ENV),
            env::var(CHILD_REMOTE_ENV),
            env::var(CHILD_BRANCH_ENV),
            env::var(CHILD_READY_ENV),
        ) else {
            return;
        };

        let _lock = acquire(&Workspace::new(root), &remote_name, &branch_name)
            .expect("child should acquire the push lock");
        fs::write(ready_path, b"locked").expect("should signal child lock readiness");
        let mut release = String::new();
        std::io::stdin()
            .read_line(&mut release)
            .expect("should wait for parent release");
    }

    #[test]
    fn test_child_process_contention_scopes_and_nested_workspace() {
        let temp = tempdir().expect("should create push lock test directory");
        let root = temp.path();
        let workspace = Workspace::new(root);
        let gen_dir = workspace.ensure_gen_dir();
        let nested_path = root.join("nested").join("deep");
        fs::create_dir_all(&nested_path).expect("should create nested workspace path");
        let nested_workspace = Workspace::new(nested_path);

        let holder = LockHolder::spawn(root, "origin", "main");
        let same_scope = acquire(&workspace, "origin", "main");
        assert!(
            matches!(same_scope, Err(RemotePushError::PushLockBusy { .. })),
            "another process must not reuse the active remote-operation session"
        );
        let nested_scope = acquire(&nested_workspace, "origin", "main");
        assert!(
            matches!(nested_scope, Err(RemotePushError::PushLockBusy { .. })),
            "nested commands should resolve to the same .gen lock"
        );

        let other_remote = acquire(&workspace, "origin2", "main")
            .expect("a different configured remote should have an independent lock");
        let other_branch = acquire(&workspace, "origin", "feature")
            .expect("a different branch should have an independent lock");
        drop(other_remote);
        drop(other_branch);

        holder.release();
        let path = lock_file_path(&gen_dir, "origin", "main");
        assert!(path.exists(), "the lock file should remain after release");
        let reacquired = acquire(&workspace, "origin", "main")
            .expect("a completed process should release its OS lock");
        drop(reacquired);
        assert!(
            path.exists(),
            "releasing ownership must not unlink the lock file"
        );
    }

    #[test]
    fn test_child_process_crash_releases_lock_without_removing_file() {
        let temp = tempdir().expect("should create push lock crash test directory");
        let workspace = Workspace::new(temp.path());
        let gen_dir = workspace.ensure_gen_dir();
        let holder = LockHolder::spawn(temp.path(), "origin", "main");
        let path = lock_file_path(&gen_dir, "origin", "main");
        assert!(
            path.exists(),
            "lock holder should create its persistent lock file"
        );

        holder.terminate();

        assert!(
            path.exists(),
            "crash recovery should keep the persistent file"
        );
        let reacquired = acquire(&workspace, "origin", "main")
            .expect("process termination should release its OS lock");
        drop(reacquired);
    }
}
