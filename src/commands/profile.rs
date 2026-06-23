use std::{error::Error, process::Command as ProcessCommand};

use clap::Args;

#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Use sampling instead of tracing spans
    #[arg(long, action)]
    pub sample: bool,
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 1..)]
    pub command: Vec<String>,
}

pub fn execute(command: Command) -> Result<(), Box<dyn Error>> {
    let executable = std::env::current_exe()?;
    let mut child = ProcessCommand::new(executable);
    child.args(&command.command).env("GEN_PROFILE", "1");
    if command.sample {
        child.env("GEN_PROFILE_SAMPLE", "1");
    }
    let status = child.status()?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("profiled command exited with status {status}").into())
    }
}
