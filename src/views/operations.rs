// Temporary stub module to keep build passing while operations view is disabled
use std::io;

use gen_models::{db::DbContext, operations::Operation};

pub fn view_operations(_ctx: &DbContext, _operations: &[Operation]) -> Result<(), io::Error> {
    // No-op placeholder
    println!("Interactive operations view is temporarily disabled.");
    Ok(())
}
