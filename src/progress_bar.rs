#[cfg(not(target_os = "emscripten"))]
use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub fn get_handler() -> MultiProgress {
    let p = MultiProgress::new();
    #[cfg(test)]
    {
        use indicatif::ProgressDrawTarget;
        p.set_draw_target(ProgressDrawTarget::hidden())
    }
    p
}

pub fn get_progress_bar(length: impl Into<Option<u64>>) -> ProgressBar {
    let length = length.into();
    let bar = if let Some(l) = length {
        ProgressBar::new(l).with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:41.cyan/blue}  {human_pos:>7}/{human_len:7} {msg}",
            )
            .unwrap(),
        )
    } else {
        let tick_string = format!("{pattern}.", pattern = ". ".repeat(20));
        let indicator = std::char::from_u32(0x0001F9EC).unwrap();
        let mut tick_strings = vec![];
        for i in (0..tick_string.len()).step_by(2) {
            if i == tick_string.len() - 1 {
                tick_strings.push(format!(
                    "{left}{indicator}",
                    left = &tick_string[..tick_string.len() - 1]
                ));
            } else {
                tick_strings.push(format!(
                    "{left}{indicator} {right}",
                    left = &tick_string[..(i + 2)],
                    right = &tick_string[(i + 2)..]
                ));
            }
        }
        ProgressBar::no_length().with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {spinner:40.cyan/blue} {human_pos:>7}{'':8} {msg}",
            )
            .unwrap()
            .tick_strings(
                &tick_strings
                    .iter()
                    .map(|f| f.as_str())
                    .collect::<Vec<&str>>(),
            ),
        )
    };
    // no thread spawning in wasm
    #[cfg(not(target_os = "emscripten"))]
    bar.enable_steady_tick(Duration::from_millis(250));
    bar
}

pub fn get_time_elapsed_bar() -> ProgressBar {
    let bar = ProgressBar::no_length().with_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {'':19}{spinner:2.cyan/blue}{'':37} {msg}",
        )
        .unwrap(),
    );
    // no thread spawning in wasm
    #[cfg(not(target_os = "emscripten"))]
    bar.enable_steady_tick(Duration::from_millis(250));
    bar
}

pub fn get_message_bar() -> ProgressBar {
    // This is a bar solely meant for giving messages to the user since the use of print causes
    // issues with indicatif
    let bar = ProgressBar::no_length().with_style(ProgressStyle::with_template("{msg}").unwrap());
    // no thread spawning in wasm
    #[cfg(not(target_os = "emscripten"))]
    bar.enable_steady_tick(Duration::from_millis(250));
    bar
}

pub fn add_saving_operation_bar(progress_bar: &MultiProgress) -> ProgressBar {
    // we have this pattern here because calling set_message before a bar is added to the multibar
    // causes duplicate lines to appear
    let bar = progress_bar.add(get_time_elapsed_bar());
    bar.set_message("Saving operation");
    bar
}
